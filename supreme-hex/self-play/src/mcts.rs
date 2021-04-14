use log::debug;
use ordered_float::NotNan;
use rand::distributions::Distribution;
use rand::prelude::*;
use statrs::distribution::Dirichlet;
use std::{fmt::Debug, hint::unreachable_unchecked};

use crate::{
    hex::{self, Hex, HexRepr, Player, Position},
    runtime::Evaluation,
};

/// A leaf evaluator. Takes in a `state` and performs some evaluation of it.
pub trait LeafEvaluator<R: HexRepr> {
    /// Creates a new leaf evaluator
    fn new() -> Self;
    /// Initialize the leaf evaluator with the state of the leaf node
    fn init(&mut self, state: Hex<R>);
    /// Called once after init. After this, the MCTS tree will repeatedly call `resume`
    /// until the leaf evaluator yields an item.
    fn start(&mut self) -> Option<f64>;
    /// Resume evaluation.
    fn resume(&mut self, evaluation: &Evaluation) -> Option<f64>;
    /// Deinitialize the leaf evaluator.
    /// It is assumed that the state returned is equivalent to the one given in `init`
    fn deinit(&mut self) -> Hex<R>;
    /// Retrieve the inner state
    fn state_mut(&mut self) -> Option<&mut Hex<R>>;
}

#[derive(Debug)]
pub struct Rollout<R: HexRepr> {
    /// The path that was taken during rollout. Needed such that we can revert it in `deinit`
    path: Vec<Action>,
    /// The `state` that is being rolled-out.
    state: Option<hex::Hex<R>>,
}

impl<R: HexRepr> LeafEvaluator<R> for Rollout<R> {
    fn new() -> Self {
        Rollout {
            path: Vec::new(),
            state: None,
        }
    }

    fn init(&mut self, state: Hex<R>) {
        debug!("init rollout p1 = {} p2 = {}", state.player1, state.player2);
        self.state = Some(state);
    }

    fn start(&mut self) -> Option<f64> {
        // Note: will only ever be called after `init`, at which point `self.state` is guaranteed to contain a value
        let state = self.state.as_ref().unwrap();
        state.winner().map(|player| match player == state.current {
            true => 1.0,
            false => -1.0,
        })
    }

    fn resume(&mut self, evaluation: &Evaluation) -> Option<f64> {
        // We yield during `start`, which means that the runtime will have performed an evaluation of `self.state` at this point.
        let state = self.state.as_mut().expect("some by construction");
        let actions = state.available_actions().collect::<Vec<_>>();
        let current = state.current;

        // Choose the action based on the policy distribution
        let action = match actions.choose_weighted(&mut rand::thread_rng(), |p| {
            evaluation.policy.get(state.size, p, &current)
        }) {
            Ok(x) => *x,
            Err(e) => panic!("choice error {:?}", e),
        };

        debug!("rollout chose action {:?}", (action.x, action.y));

        // Perform the action, and record it for later unwinding
        state.place(action);
        self.path.push(action);

        // If this is a terminal state, we will return the end-result
        let path_len = self.path.len();
        state.winner().map(|winner| {
            let root = match path_len % 2 {
                0 => state.current,
                1 => state.current.next(),
                // Safety: x % 2 is always either in {0, 1} when x >= 0,
                // which it is in this case (since self.path.len() is an usize)
                _ => unsafe { unreachable_unchecked() },
            };

            debug!(
                "terminal state with p1 = {}, p2 = {}",
                state.player1, state.player2
            );

            match winner == root {
                true => 1.0,
                false => -1.0,
            }
        })
    }

    fn deinit(&mut self) -> Hex<R> {
        debug!("deinitializing rollout, will undo: {:?}", &self.path);

        // We will hand the state back to the MCTS tree
        let mut state = self.state.take().expect("some by construction");
        // Unwind the actions done in the rollout
        for position in self.path.drain(..).rev() {
            state.unplace(position);
        }

        state
    }

    fn state_mut(&mut self) -> Option<&mut Hex<R>> {
        self.state.as_mut()
    }
}

pub struct ValueFunction<R: HexRepr> {
    state: Option<Hex<R>>,
}

impl<R: HexRepr> LeafEvaluator<R> for ValueFunction<R> {
    fn new() -> Self {
        ValueFunction { state: None }
    }

    fn init(&mut self, state: Hex<R>) {
        self.state = Some(state);
    }

    fn start(&mut self) -> Option<f64> {
        // Note: will only ever be called after `init`, at which point `self.state` is guaranteed to contain a value
        let state = self.state.as_ref().unwrap();
        state.winner().map(|player| match player == state.current {
            true => 1.0,
            false => -1.0,
        })
    }

    fn resume(&mut self, evaluation: &Evaluation) -> Option<f64> {
        Some(evaluation.value.get() as f64)
    }

    fn deinit(&mut self) -> Hex<R> {
        self.state.take().expect("should exist")
    }

    fn state_mut(&mut self) -> Option<&mut Hex<R>> {
        self.state.as_mut()
    }
}

#[derive(Debug)]
pub struct Sample {
    // The state of Hex that was present
    pub state: Hex<()>,
    // The resulting probability distribution after Monte Carlo simulations (i.e. share of visit counts)
    pub distribution: Vec<(Action, f64)>,
    // The winning player at the end of the episode
    pub winner: Player,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandMode {
    WithNoise,
    WithoutNoise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Status representing the case where the MCTS is awaiting evaluation of a leaf node, such that it can
    /// update all edges with the prior probabilities
    AwaitingExpansion(ExpandMode),
    /// Representing the case where the MCTS tree is awaiting evaluation during rollouts
    AwaitingLeafEvaluation,
    /// Representing the case where an episode was just finished
    FinishedEpisode,
    /// The case where a simulation was just finished
    FinishedSimulation,
}

type Action = hex::Position;

#[derive(Debug, Clone, Copy)]
pub struct Edge {
    /// Action values. This is simply w / n, i.e. the mean action value of this state
    /// over all simulations.
    q: f64,
    /// The accumulated reward over simulations utilizing this edge.
    w: f64,
    /// The total number of times thie edge has been traversed
    n: u64,
    /// The prior probability of this edge
    p: f64,
    /// The action that was performed
    action: Action,
    /// The child node along this edge. Given as an index into the MCTS' list.
    child: usize,
}

#[derive(Debug)]
pub struct Node {
    /// The index of this node's parent. The initial root node has a reference to itself (i.e. = 0)
    parent: usize,
    /// The child nodes of this node. `children[i]` is the child when applying action `i`, given as an index into the MCTS' list.
    edges: Vec<Edge>,
    /// The sum of visit counts of its children
    n: u64,
}

impl Node {
    pub fn new(parent: usize) -> Self {
        Node {
            parent,
            edges: Vec::new(),
            n: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PolicyKind {
    Proportionate,
    Greedy,
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Termination criteria
    pub simulations: usize,
    /// Chooses weighting between Q value and U value
    pub c: f64,
    /// The epsilon value used to weight the Dirichlet noise vs the prior probabilities in the root node.
    /// More specifically, we use p = (1-epsilon) * prior + epsilon * noise.
    pub epsilon: f64,
    /// Whether self.policy should be proportionate or greedy w.r.t visit counts.
    pub policy_kind: PolicyKind,
}

/// Implements a resumable Monte Carlo Tree search. Purpose is to enable concurrent expansion of several MCTS's
/// at once, such that input to tensorflow can be batched
#[derive(Debug)]
pub struct MCTS<R: HexRepr + Debug, L: LeafEvaluator<R>> {
    //  CONFIGURATION
    //  ----------------------------------
    /// The game we're mutating. Note that we use a single game that is mutated going down the tree, with actions being undone while traversing up again
    pub state: Option<hex::Hex<R>>,
    /// The leaf evaluator utilized
    pub leaf_evaluator: L,
    /// The configuration
    pub config: Config,

    //  STATE VARIABLES
    //  ----------------------------------
    /// The state of the tree search. This is used to keep track of where to resume from
    /// after having yielded control to the runtime
    pub status: Status,
    /// The current root node, given as an index into `self.nodes`
    pub root: usize,
    /// The currently active node, given as an index into `self.nodes`
    pub current: usize,
    /// The number of simulations done.
    pub simulations_done: usize,
    /// To make the borrow checker happy, we will store the nodes in a flat list for each episode.
    /// Even though this makes it impossible to discard parts of the search tree as we go downwards, it should not
    /// be a large problem for this game.
    pub nodes: Vec<Node>,
    /// The indices of the edges (relative to the node's set of available actions)
    /// that were traversed during the in-tree traversal
    pub in_tree_path: Vec<usize>,
    /// The nodes visited during self-play. Will form the base of the samples
    /// at the end of an episode
    pub visited_nodes: Vec<(Option<Position>, hex::Hex<()>, usize)>,
    /// The distribution from which we will draw noise for priors
    pub dirichlet: Dirichlet,
}

impl<R: HexRepr + Debug, L: LeafEvaluator<R>> MCTS<R, L> {
    pub fn new(state: hex::Hex<R>, leaf_evaluator: L, config: Config) -> Self {
        MCTS {
            dirichlet: Dirichlet::new_with_param(
                10.0 / (0.75 * (state.size * state.size) as f64),
                state.size * state.size,
            )
            .unwrap(),
            visited_nodes: Vec::new(),
            state: Some(state),
            config,
            leaf_evaluator,
            status: Status::FinishedEpisode,
            root: 0,
            current: 0,
            simulations_done: 0,
            nodes: Vec::new(),
            in_tree_path: Vec::new(),
        }
    }

    /// The U-value of the PUCT
    fn u(&self, node: &Node, edge: &Edge) -> f64 {
        edge.p * (node.n as f64).sqrt() / (1.0 + edge.n as f64)
    }

    /// The action according to PUCT. Returns none if the node has no children,
    /// which is equivalent to it being a child node.
    fn puct_policy<'a>(&self, node: &'a Node) -> Option<(usize, &'a Edge)> {
        node.edges
            .iter()
            .enumerate()
            // Safety: by construction, we known that neither Q nor U are NaN.
            .max_by_key(|(_, edge)| unsafe {
                NotNan::unchecked_new(edge.q + self.config.c * self.u(node, edge))
            })
    }

    /// Resume the search from the last yield point. Continues until the next yield point
    /// We can summarize the search as follows:
    /// yield    what                                 | note
    ///       1. Initialize root node.                | done after an episode has ended (and when first resuming)
    ///       2. In-tree-traversal until leaf node.   | must keep track of path
    ///   *   3. Expand leaf node                     |
    ///   *   4. Rollout                              | must keep track of current state in rollout
    ///   *   5. Backtrack                            | yield Vec<Sample>s
    ///       6. Go to 2
    ///
    pub fn resume(&mut self, evaluation: Option<&Evaluation>) -> Option<Vec<Sample>> {
        debug!("resuming with status = {:?}", self.status);

        match self.status {
            // Assumption: the asked for model inference has been done
            Status::AwaitingExpansion(mode) => {
                self.expand(
                    mode,
                    evaluation.expect("evaluation available when resuming at AwaitingExpansion"),
                );
                self.start_evaluation()
            }
            // Asumption: the inference has been done.
            Status::AwaitingLeafEvaluation => self.resume_evaluation(
                evaluation.expect("evaluation available when resuming at AwaitingLeafEvaluation"),
            ),
            Status::FinishedEpisode => {
                // TODO: add Dirichlet noise to root node?
                self.initialize_root();
                self.intree_traversal();
                self.start_expansion(ExpandMode::WithNoise)
            }
            Status::FinishedSimulation => {
                self.simulations_done += 1;

                if self.simulations_done <= self.config.simulations {
                    // Reset back to the root node
                    self.current = self.root;
                    self.intree_traversal();
                    self.start_expansion(ExpandMode::WithoutNoise)
                } else {
                    self.execute_step(None)
                }
            }
        }
    }

    pub fn root_policy(&self) -> Vec<(Position, f64)> {
        let root = &self.nodes[0];
        root.edges
            .iter()
            .map(|e| (e.action, (e.n as f64) / (root.n as f64)))
            .collect()
    }

    pub fn current_policy(&self) -> Vec<(Position, f64)> {
        let root = &self.nodes[self.root];
        root.edges
            .iter()
            .map(|e| (e.action, (e.n as f64) / (root.n as f64)))
            .collect()
    }

    fn finish_episode(&mut self, winner: Player) -> Option<Vec<Sample>> {
        debug!("finish_episode(winner: {:?})", winner);
        self.status = Status::FinishedEpisode;

        let mut mcts_state = self.state.take().expect("should be some");

        let samples = Some(
            self.visited_nodes
                .iter()
                .rev()
                .map(|(action, state, i)| {
                    // Undo the action, if any
                    match action {
                        Some(position) => mcts_state.unplace(*position),
                        None => (),
                    };

                    let node = &self.nodes[*i];
                    Sample {
                        // Note: copy to keep the borrow checker happy. Is cheap to copy anyways.
                        state: state.minimal(),
                        distribution: node
                            .edges
                            .iter()
                            .map(|e| (e.action, (e.n as f64) / (node.n as f64)))
                            .collect(),
                        winner,
                    }
                })
                .collect(),
        );

        debug!(
            "end of finish_episode has state p1 = {}, p2 = {}",
            mcts_state.player1, mcts_state.player2
        );

        self.state = Some(mcts_state);

        samples
    }

    pub fn execute_step(&mut self, action: Option<Position>) -> Option<Vec<Sample>> {
        let edge = match action {
            None => {
                let edge = self.policy(&self.nodes[self.current]);
                match edge {
                    Some(e) => e,
                    None => panic!("{:?}", &self.nodes[self.current]),
                }
            }
            Some(action) => {
                let node = &self.nodes[self.root];
                match node.edges.iter().find(|x| x.action == action) {
                    Some(e) => e,
                    None => panic!("{:?} is not part of the edges {:?}", action, &node.edges),
                }
            }
        };

        let state = self.state.as_mut().unwrap();

        // Do all book keeping necessary to rebase the root node to `edge.child`
        self.root = edge.child;
        self.current = self.root;
        //self.current = edge.child;
        self.simulations_done = 0;
        state.place(edge.action);

        debug!("execute_step, visited_nodes = {:?}", self.visited_nodes);
        // Now, if we're in a terminal state, it means we've won, and can collect our samples.
        // If not, we such prepare for a new simulation from the rebased root node
        match state.winner() {
            Some(winner) => {
                state.unplace(edge.action);
                self.finish_episode(winner)
            }
            None => {
                self.status = Status::FinishedSimulation;
                // This must be after the check for the winner, since we have no visit counts for
                // a terminal nodes (yielding NaNs for the policy).
                self.visited_nodes
                    .push((Some(edge.action), state.minimal(), self.current));
                None
            }
        }
    }

    /// Chooses an action based on the visit counts in the current node
    fn policy<'a>(&self, node: &'a Node) -> Option<&'a Edge> {
        node.edges
            .choose_weighted(&mut rand::thread_rng(), |e| e.n)
            .ok()
    }

    // Initialize the root for a new episode.
    fn initialize_root(&mut self) {
        debug!(
            "initialize root. state = {:?}",
            self.state.as_ref().unwrap().minimal()
        );
        // Clear nodes from the previous episode.
        self.nodes.clear();
        // Clear the path from the previous episode
        self.visited_nodes.clear();
        self.visited_nodes
            .push((None, self.state.as_ref().unwrap().minimal(), 0));
        // Add an initial root node
        self.nodes.push(Node::new(0));
        // Set the root node as active
        self.current = 0;
        // Set the root node as the root
        self.root = 0;
    }

    // Perform the in-tree traversel according to the PUCT policy. Updates `self.current` at each step.
    fn intree_traversal(&mut self) {
        self.in_tree_path.clear();
        // This is to make the borrow checker happy
        let mut state = self.state.take().expect("should be some");
        loop {
            match self.puct_policy(&self.nodes[self.current]) {
                Some((i, edge)) => {
                    state.place(edge.action);
                    self.in_tree_path.push(i);
                    self.current = edge.child;
                }
                None => break,
            }
        }
        self.state = Some(state);
    }

    fn backpropagate(&mut self, mut z: f64) -> Option<Vec<Sample>> {
        // The state is known to be non-empty by construction. It is only ever `None` while control is given to the leaf evaluator
        let state = self
            .state
            .as_mut()
            .expect("`state` should be non-empty (by construction)");

        debug!(
            "backpropagating {} moves from current state p1 = {}, p2 = {}",
            self.in_tree_path.len(),
            state.player1,
            state.player2
        );
        // We are currently at a just-expanded leaf node, which means we have no Monte-Carlo samples
        while let Some(edge) = self.in_tree_path.pop() {
            // Since we assume a zero-sum game, the reward of a the previous player is the negative
            // of that of the current player
            z = -z;

            // Backtrack one step up, and update
            self.current = self.nodes[self.current].parent;
            let node = &mut self.nodes[self.current];
            let edge = &mut node.edges[edge];
            // Undo the action performed along `edge`
            state.unplace(edge.action);

            // Update visit W and N values
            edge.w += z;
            edge.n += 1;
            edge.q = (edge.w as f64) / (edge.n as f64);
            node.n += 1;
        }

        // This should hold, by construction.
        assert!(self.current == self.root);

        // After back-propagation, we're set of another round of simulations
        self.status = Status::FinishedSimulation;

        None
    }

    fn start_evaluation(&mut self) -> Option<Vec<Sample>> {
        // Initialize the leaf evaluator and set the status to signify that we have passed control over to it.
        let state = self.state.take().expect("non-empty by construction");

        debug!(
            "starting evaluation of p1 = {}, p2 = {}",
            state.player1, state.player2
        );

        self.leaf_evaluator.init(state);
        self.status = Status::AwaitingLeafEvaluation;

        // Start evaluation of the leaf state.
        match self.leaf_evaluator.start() {
            Some(z) => {
                self.state = Some(self.leaf_evaluator.deinit());
                self.backpropagate(z)
            }
            None => None,
        }
    }

    fn start_expansion(&mut self, mode: ExpandMode) -> Option<Vec<Sample>> {
        // We will simply set the state as awaiting expansion, and pass control back to the runtime
        self.status = Status::AwaitingExpansion(mode);
        None
    }

    fn resume_evaluation(&mut self, evaluation: &Evaluation) -> Option<Vec<Sample>> {
        // At this point we have earlier submitted a state for evaluation, which should be ready now.
        match self.leaf_evaluator.resume(evaluation) {
            Some(z) => {
                self.state = Some(self.leaf_evaluator.deinit());
                self.backpropagate(z)
            }
            None => None,
        }
    }

    fn expand(&mut self, mode: ExpandMode, evaluation: &Evaluation) {
        let state = self
            .state
            .as_ref()
            .expect("state should be available in `expand`");

        debug!(
            "expanding p1 = {}, p2 = {}. {} moves available",
            state.player1,
            state.player2,
            state.available_actions().count()
        );

        let current_idx = self.current;
        assert!(self.nodes[current_idx].edges.is_empty());

        let dirichlet = match mode {
            ExpandMode::WithNoise => self.dirichlet.sample(&mut rand::thread_rng()),
            ExpandMode::WithoutNoise => Vec::new(),
        };

        for (i, action) in state.available_actions().enumerate() {
            // Insert a new node into the MCTS tree
            self.nodes.push(Node::new(current_idx));
            // Retrieve its index, such that we can wire up the edges from `current`
            let new_idx = self.nodes.len() - 1;
            // This is the prior probability:
            let prior = evaluation.policy.get(state.size, &action, &state.current) as f64;
            let noise = match mode {
                ExpandMode::WithNoise => dirichlet[i],
                ExpandMode::WithoutNoise => 0.0,
            };
            let epsilon = match mode {
                ExpandMode::WithNoise => self.config.epsilon,
                ExpandMode::WithoutNoise => 0.0,
            };

            let edge = Edge {
                q: 0.0,
                w: 0.0,
                n: 0,
                p: (1.0 - epsilon) * prior + epsilon * noise,
                action,
                child: new_idx,
            };

            self.nodes[current_idx].edges.push(edge);
        }
    }
}
