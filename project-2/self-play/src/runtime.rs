use std::{cell::UnsafeCell, fmt::Debug, marker::PhantomData, ops::Range};

use log::{debug, error};
use tensorflow as tf;
use tf::{OutputName, Status};

use crate::{
    encoder::Encoder,
    hex::{Hex, Player, Position},
    mcts::{self, LeafEvaluator, Sample, MCTS},
};

/// The runtime responsible for orchestrating control flow between tensorflow and the MCTS
#[derive(Debug)]
pub struct Runtime<E: Encoder + Debug, L: LeafEvaluator<E>> {
    /// The tensorflow graph
    pub graph: tf::Graph,
    /// The meta graph definition
    pub meta_graph: tf::MetaGraphDef,
    /// The tensorflow session
    pub session: tf::Session,
    /// The input batch
    pub input_batch: UnsafeCell<tf::Tensor<f32>>,
    /// The concurrent tree searches we're executing
    pub tree_searches: Vec<MCTS<E, L>>,
    /// The game state that will be the topmost root.
    pub root: Hex<()>,
    /// The input name
    input_name: OutputName,
    /// The policy name
    policy_name: OutputName,
    /// The value name
    value_name: OutputName,
    /// A phantom type to fix the encoder
    encoder_marker: PhantomData<E>,
}

#[derive(Debug)]
pub enum InitError {
    Failed,
    NoShape,
    InvalidInputShape {
        expected: Vec<Option<i64>>,
        actual: Vec<Option<i64>>,
    },
    FixedBatchSize,
    Status(Status),
}

impl From<Status> for InitError {
    fn from(status: Status) -> Self {
        InitError::Status(status)
    }
}

#[derive(Debug)]
pub struct BatchItem {
    pub dims: Vec<u64>,
    pub data: Range<*mut f32>,
}

impl BatchItem {
    /// Zeroes all entries in this batch item
    pub fn zero(&mut self) {
        let mut start = self.data.start;
        let end = self.data.end;

        while start != end {
            unsafe { start.write(0.0) }

            start = unsafe { start.offset(1) }
        }
    }

    pub fn set(&mut self, indices: &[u64], value: f32) {
        let index = self.get_index(indices);
        let start = self.data.start;

        // Safety: self.get_index will panic if given an invalid index
        unsafe {
            start.offset(index as isize).write(value);
        }
    }

    pub fn get(&self, indices: &[u64]) -> f32 {
        let index = self.get_index(indices);
        let start = self.data.start;

        // Safety: self.get_index will panic if given an invalid index
        unsafe { start.offset(index as isize).read() }
    }
    /// Copied straight from the tensorflow source:)
    /// Get the array index from rows / columns indices.
    ///
    /// ```
    /// # use tensorflow::Tensor;
    /// let a = BatchItem { dims: Vec::new(), data: (std::ptr::null_mut()..std::ptr::null_mut())};
    ///
    /// assert_eq!(a.get_index(&[2, 2, 2]), 26);
    /// assert_eq!(a.get_index(&[1, 2, 2]), 17);
    /// assert_eq!(a.get_index(&[1, 2, 0]), 15);
    /// assert_eq!(a.get_index(&[1, 0, 1]), 10);
    /// ```
    pub fn get_index(&self, indices: &[u64]) -> usize {
        assert!(
            self.dims.len() == indices.len(),
            "length check failed: dims = {:?}, indices = {:?}",
            self.dims,
            indices
        );
        let mut index = 0;
        let mut d = 1;
        for i in (0..indices.len()).rev() {
            assert!(self.dims[i] > indices[i]);
            index += indices[i] * d;
            d *= self.dims[i];
        }
        index as usize
    }
}

pub struct Evaluation {
    pub policy: PolicyEvaluation,
    pub value: ValueEvaluation,
}

pub struct PolicyEvaluation {
    inner: BatchItem,
}

impl PolicyEvaluation {
    pub fn get(&self, size: usize, action: &Position, player: &Player) -> f32 {
        self.inner
            .get(&[Self::get_index(size, action, player) as u64])
    }

    /// TODO: Generalize
    pub fn get_index(size: usize, action: &Position, player: &Player) -> usize {
        match player {
            Player::One => action.y + size * action.x,
            Player::Two => action.x + size * action.y,
        }
    }
}

pub struct ValueEvaluation {
    inner: BatchItem,
}

impl ValueEvaluation {
    pub fn get(&self) -> f32 {
        self.inner.get(&[0])
    }
}

impl<E, L> Runtime<E, L>
where
    E: Encoder + Debug,
    L: LeafEvaluator<E>,
{
    fn validate_input_shape(
        graph: &tf::Graph,
        meta_graph: &tf::MetaGraphDef,
        root: &Hex<()>,
    ) -> Result<(OutputName, Vec<u64>), InitError> {
        // Get the tensor info of the input
        let signature = meta_graph.get_signature("serving_default")?;
        let input = signature.get_input("input")?;

        // This is the input shape required by the encoder type `E`
        let encoder_shape = E::shape(root.size);

        // Extract the actual input shape from the graph
        let input_op = graph.operation_by_name_required(&input.name().name)?;
        let input_shape = graph.tensor_shape(input_op)?;

        // This is the shape required by for our encoder to work
        let expected = std::iter::once(None)
            .chain(encoder_shape.iter().map(|x| Some(*x as i64)))
            .collect();
        // The actual input shape
        let actual = (0..input_shape.dims().ok_or(InitError::Failed)?)
            .map(|i| input_shape[i])
            .collect();

        if actual != expected {
            return Err(InitError::InvalidInputShape { expected, actual });
        }

        Ok((input.name().clone(), encoder_shape))
    }

    pub fn validate_output_shapes(
        meta_graph: &tf::MetaGraphDef,
    ) -> Result<(OutputName, OutputName), InitError> {
        let signature = meta_graph.get_signature("serving_default")?;
        let policy = signature.get_output("policy")?.name().clone();
        let value = signature.get_output("value")?.name().clone();
        // TODO: validate
        Ok((policy, value))
    }

    pub fn policy_distibution(
        size: usize,
        states: &[Hex<()>],
        graph: tf::Graph,
        meta_graph: tf::MetaGraphDef,
        session: tf::Session,
        config: mcts::Config,
    ) -> Result<Vec<Vec<((usize, usize), f64)>>, InitError> {
        let mut runtime = Self::with_states(size, states, graph, meta_graph, session, config)?;

        let mut moves = vec![None; states.len()];
        let mut remaining = states.len();

        while remaining > 0 {
            // Prepare all inner states for evaluation
            for mcts in runtime.tree_searches.iter_mut() {
                match mcts.state.as_mut() {
                    Some(state) => state.inner.finalize(),
                    None => (),
                }

                match mcts.leaf_evaluator.state_mut() {
                    Some(state) => state.inner.finalize(),
                    None => (),
                }
            }
            // Perform evaluation. _p, and _v are kept such that the inner tensor doesn't go out of scope.
            let (_p, _v, evaluations) = runtime.evaluate().expect("evaluation should not fail");

            for (i, (mcts, evaluation)) in runtime
                .tree_searches
                .iter_mut()
                .zip(evaluations)
                .enumerate()
            {
                // We're interested in the policy from the root node
                if mcts.simulations_done == mcts.config.simulations && moves[i].is_none() {
                    remaining -= 1;
                    moves[i] = Some(mcts.root_policy());
                }

                let _ = mcts.resume(Some(&evaluation));
            }
        }

        Ok(moves
            .into_iter()
            .map(|p| p.expect("should exist"))
            .map(|xs| xs.into_iter().map(|(p, x)| ((p.x, p.y), x)).collect())
            .collect())
    }

    pub fn with_states(
        size: usize,
        states: &[Hex<()>],
        graph: tf::Graph,
        meta_graph: tf::MetaGraphDef,
        session: tf::Session,
        config: mcts::Config,
    ) -> Result<Self, InitError> {
        let root = Hex::empty(size, ());
        let (input_name, mut shape) = Self::validate_input_shape(&graph, &meta_graph, &root)?;
        let (policy_name, value_name) = Self::validate_output_shapes(&meta_graph)?;
        let batch_size = states.len();
        // We want to fix the batch size to `concurrents`
        shape.insert(0, batch_size as u64);

        let input_batch = UnsafeCell::new(tf::Tensor::<f32>::new(&shape[..]));
        let inputs = unsafe { input_batch.split_batch().expect("non-null by construction") };

        assert!(inputs.len() == batch_size);

        let mut tree_searches = inputs
            .into_iter()
            .map(|inner| MCTS::new(root.with_inner(E::new(inner)), L::new(), config.clone()))
            .collect::<Vec<_>>();

        for (root, mcts) in states.iter().zip(&mut tree_searches) {
            assert!(root.size == size);
            let mut placed1 = Vec::new();
            let mut placed2 = Vec::new();
            let state = mcts.state.as_mut().unwrap();
            for x in 0..size {
                for y in 0..size {
                    let position = Position { x, y };

                    if root.is_occupied(position, Player::One) {
                        placed1.push(position);
                    }

                    if root.is_occupied(position, Player::Two) {
                        placed2.push(position);
                    }
                }
            }

            assert!(placed1.len() == placed2.len() || placed1.len() == placed2.len() + 1);
            let mut it1 = placed1.into_iter();
            let mut it2 = placed2.into_iter();

            loop {
                match (it1.next(), it2.next()) {
                    (Some(a), Some(b)) => {
                        state.place(a);
                        state.place(b);
                    }
                    (Some(a), _) => state.place(a),
                    (None, None) => break,
                    _ => unreachable!("this should not happen"),
                }
            }
        }

        Ok(Runtime {
            input_name,
            policy_name,
            value_name,
            graph,
            meta_graph,
            session,
            input_batch,
            tree_searches,
            root,
            encoder_marker: PhantomData,
        })
    }

    pub fn new(
        graph: tf::Graph,
        meta_graph: tf::MetaGraphDef,
        session: tf::Session,
        root: Hex<()>,
        concurrents: usize,
        config: mcts::Config,
    ) -> Result<Self, InitError> {
        let (input_name, mut shape) = Self::validate_input_shape(&graph, &meta_graph, &root)?;
        let (policy_name, value_name) = Self::validate_output_shapes(&meta_graph)?;
        let batch_size = concurrents;
        // We want to fix the batch size to `concurrents`
        shape.insert(0, batch_size as u64);

        let input_batch = UnsafeCell::new(tf::Tensor::<f32>::new(&shape[..]));
        let inputs = unsafe { input_batch.split_batch().expect("non-null by construction") };

        assert!(inputs.len() == batch_size);

        let tree_searches = inputs
            .into_iter()
            .map(|inner| MCTS::new(root.with_inner(E::new(inner)), L::new(), config.clone()))
            .collect();

        Ok(Runtime {
            input_name,
            policy_name,
            value_name,
            graph,
            meta_graph,
            session,
            input_batch,
            tree_searches,
            root,
            encoder_marker: PhantomData,
        })
    }

    fn evaluate(
        &mut self,
    ) -> Result<
        (
            UnsafeCell<tf::Tensor<f32>>,
            UnsafeCell<tf::Tensor<f32>>,
            Vec<Evaluation>,
        ),
        Status,
    > {
        let mut run = tf::SessionRunArgs::new();

        let input_op = self
            .graph
            .operation_by_name_required(&self.input_name.name)?;
        run.add_feed(&input_op, 0, unsafe { &(*self.input_batch.get()) });

        // Note: we assume that policy and value belong to the same output
        let op_policy = self
            .graph
            .operation_by_name_required(&self.policy_name.name)?;
        let op_value = self
            .graph
            .operation_by_name_required(&self.value_name.name)?;

        run.add_target(&op_policy);
        run.add_target(&op_value);
        let policy_token = run.request_fetch(&op_policy, self.policy_name.index);
        let value_token = run.request_fetch(&op_value, self.value_name.index);

        self.session.run(&mut run)?;

        let policy = UnsafeCell::new(run.fetch::<f32>(policy_token)?);
        let value = UnsafeCell::new(run.fetch::<f32>(value_token)?);

        let policies = unsafe { policy.split_batch().unwrap() };
        let values = unsafe { value.split_batch().unwrap() };

        assert!(policies.len() == values.len());

        let evaluations = policies
            .into_iter()
            .zip(values)
            .map(|(policy, value)| Evaluation {
                policy: PolicyEvaluation { inner: policy },
                value: ValueEvaluation { inner: value },
            })
            .collect();

        Ok((policy, value, evaluations))
    }

    pub fn generate(&mut self, samples: usize) -> Vec<Sample> {
        let mut generated = Vec::new();

        while generated.len() < samples {
            // Prepare all inner states for evaluation
            for mcts in self.tree_searches.iter_mut() {
                match mcts.state.as_mut() {
                    Some(state) => state.inner.finalize(),
                    None => (),
                }

                match mcts.leaf_evaluator.state_mut() {
                    Some(state) => state.inner.finalize(),
                    None => (),
                }
            }
            // Perform evaluation. _p, and _v are kept such that the inner tensor doesn't go out of scope.
            let (_p, _v, evaluations) = self.evaluate().expect("evaluation should not fail");

            for (mcts, evaluation) in self.tree_searches.iter_mut().zip(evaluations) {
                match mcts.resume(Some(&evaluation)) {
                    Some(samples) => {
                        if samples.len() < self.root.size
                            || samples.len() > self.root.size * self.root.size + 1
                        {
                            error!(
                                "received impossible set of samples (len = {}): {:?}",
                                samples.len(),
                                &samples
                            );
                        }
                        debug!("received samples: {:?}", &samples);

                        generated.extend(samples)
                    }
                    None => (),
                }
            }
        }

        generated
    }
}

pub trait SplitBatch {
    unsafe fn split_batch(&self) -> Option<Vec<BatchItem>>;
}

impl SplitBatch for UnsafeCell<tf::Tensor<f32>> {
    unsafe fn split_batch(&self) -> Option<Vec<BatchItem>> {
        let inner = self.get();
        let (batch_size, dims) = {
            let (b, d) = (*inner).dims().split_first()?;
            (*b, d.to_vec())
        };

        // The size of each item in the batch
        let size = dims.iter().product::<u64>() as usize;

        let items: Vec<BatchItem> = (*inner)
            .chunks_exact_mut(size)
            .map(|data| BatchItem {
                dims: dims.clone(),
                data: data.as_mut_ptr_range(),
            })
            .collect();

        assert!(items.len() == batch_size as usize);

        Some(items)
    }
}
