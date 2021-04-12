from typing import NamedTuple, TypeVar, Generic, List, Union
from random import random, choice, choices
from itertools import count
import collections

from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Dense
from .games import Hex
import tensorflow as tf
from tensorflow.python.keras.layers.pooling import MaxPool2D
#import graphviz
import numpy as np

State = TypeVar('State')
Action = TypeVar('Action')


# @tf.function
def policy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(- y_true * tf.math.log(tf.math.maximum(y_pred, 0.0001)), axis=[1]))


class Actor:
    def policy(self, state, callback=None):
        raise NotImplementedError()

    def policies(self, states):
        return [self.policy(state) for state in states]


class SFAgent(Actor):
    """An agent relying on the Rust-implemented MCTS search"""

    def __init__(self, model_path, size, leaf_evaluation='rollout', encoder='normalized', simulations=1000, c=3, policy='greedy'):
        assert policy in ['greedy', 'proportional']

        self.model_path = model_path
        self.size = size
        self.leaf_evaluation = leaf_evaluation
        self.encoder = encoder
        self.simulations = simulations
        self.c = c
        self.policy_kind = policy

    def policies(self, states):
        #assert all(state.size == self.size for state in states)
        import self_play
        policies = self_play.policy_distribution(
            leaf_evaluation=self.leaf_evaluation,
            encoder=self.encoder,
            model_path=self.model_path,
            size=self.size,
            states=[(state.grid if isinstance(state, Hex) else state)
                    for state in states],
            simulations=self.simulations,
            c=self.c,
        )

        if self.policy_kind == 'greedy':
            return [max(policy, key=lambda t: t[1])[0] for policy in policies]
        else:
            return [choices(policy, weights=(x for (p, x) in policy), k=1)[0] for policy in policies]

    def policy(self, state, callback=None):
        #assert state.size == self.size
        return self.policies([state])[0]


class LeafEvaluator:
    def evaluate(self, state) -> float:
        raise NotImplementedError()


class Rollout(LeafEvaluator):
    """A rollout evaluator"""

    def __init__(self, policy):
        self.policy = policy

    def evaluate(self, state):
        # We assume that we only get a reward at the end of the game.
        initial = state
        while not state.is_final():
            action = self.policy.policy(state)
            state = state.next_state(action)

        winner = state.winner()
        # Tie
        if winner is None:
            return 0
        # Current player won
        elif winner == initial.current_player:
            return 1
        # Opponent won
        else:
            return -1


class Output:
    def __init__(self, P, z, size, i):
        self.P = P
        self.z = z
        self.size = size
        self.i = i

    def action(self, state, A=None):
        A = state.available_actions() if A is None else A
        return max(A, key=lambda a: self.P[self.i, a[1] * self.size + a[0]])

    def desirabilities(self, actions):
        return [self.P[self.i, y * self.size + x] for x, y in actions]


class BetaHex(tf.keras.Model, LeafEvaluator, Actor):
    def __init__(self,
                 size,
                 encoder,
                 shape,
                 optimizer=tf.keras.optimizers.Adam(),
                 ):
        super(BetaHex, self).__init__()
        # The size of the game board
        self.size = size
        # Used to convert a state representation to a tensor
        self.encoder = encoder
        # The optimizer used
        self.optimizer = optimizer

        # The main body of the network. We utilize a shared body
        # and two distinct 'heads': one for policy and one for value.
        self.body = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                256, 3, activation='swish', kernel_regularizer=tf.keras.regularizers.L2(0.0)),
            tf.keras.layers.Conv2D(
                384, 2, activation='swish', kernel_regularizer=tf.keras.regularizers.L2(0.0)),
            tf.keras.layers.Conv2D(
                512, 2, activation='swish', kernel_regularizer=tf.keras.regularizers.L2(0.0)),
        ], name='body')

        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                4 * size * size, kernel_regularizer=tf.keras.regularizers.L2(0.0), activation='swish'),
            tf.keras.layers.Dense(
                size * size, kernel_regularizer=tf.keras.regularizers.L2(0.0), activation='swish'),
        ], name='policy-head')

        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                size * size, kernel_regularizer=tf.keras.regularizers.L2(0.0), activation='swish'),
            # tf.keras.layers.Dense(1, activation='tanh',
            #                      kernel_regularizer=tf.keras.regularizers.L2(0.01))
        ], name='value-head')

        # This is our input.
        self.input_layer = tf.keras.layers.Input(shape=shape, name='input')
        self.out = self.body(self.input_layer)
        self.policy_layer = tf.keras.layers.Softmax(
            name='policy')(self.policy_head(self.out))
        self.value_layer = tf.keras.layers.Dense(
            1, activation='tanh', kernel_regularizer=tf.keras.regularizers.L2(0.0), name='value')(self.value_head(self.out))

        self.model = tf.keras.Model(inputs=self.input_layer, outputs=[
                                    self.policy_layer, self.value_layer])

        self.model.compile(
            optimizer=optimizer,
            metrics={'policy': tf.keras.metrics.CategoricalAccuracy(
            ), 'value': 'mse'},
            loss_weights={'value': 1.0, 'policy': 1.0},
            loss={'policy': policy_loss, 'value': 'mse'},
        )

    def call(self, input):
        return self.model(input)

    def desirability(self, state, action) -> float:
        (x, y) = action
        return self(tf.expand_dims(self.encoder(state), axis=0))[0].numpy()[0, y * self.size + x]

    def desirabilities(self, state, actions):
        (P, z) = self(tf.expand_dims(self.encoder(state), axis=0))
        #P = P[0].numpy()

        return [P[0, y * self.size + x] for x, y in actions]

    def evaluate(self, state):
        (policies, values) = self(tf.expand_dims(self.encoder(state), axis=0))
        return values.numpy()[0, 0]

    def policy(self, state, A=None):
        A = state.available_actions() if A is None else A
        p = self(tf.expand_dims(self.encoder(state), axis=0))[0].numpy()[0]
        return max(A, key=lambda a: p[a[1] * self.size + a[0]])

    def batch(self, states):
        batch = tf.stack([self.encoder(s) for s in states])
        (P, z) = self(batch)
        return [Output(P, z, s.size, i) for i, s in enumerate(states)]
        # return [max(s.available_actions(), key=lambda a: p[a[1] * self.size + a[0]]) for s, p in zip(states, P)]

    def batch_policy(self, states):
        batch = tf.stack([self.encoder(s) for s in states])
        (P, _) = self(batch)
        return [max(s.available_actions(), key=lambda a: p[a[1] * self.size + a[0]]) for s, p in zip(states, P)]

    def numpy_distribution(self, D):
        array = np.zeros(self.size * self.size)
        for (x, y), p in D:
            array[self.size * y + x] = p

        return tf.convert_to_tensor(array, dtype=tf.float32)


Edge = collections.namedtuple('Edge', ['W', 'N', 'P'])


class Node:
    def with_dirichlet(state, alpha, epsilon, root):
        A = list(state.available_actions())

        node = Node(state, A=A)
        priors = root.default_policy.desirabilities(state, A)
        noise = np.random.default_rng().dirichlet([alpha for _ in A])

        node.statistics = {
            a: Edge(W=0, N=0, P=(1 - epsilon) * p + epsilon * n) for a, p, n in zip(A, priors, noise)
        }

        node.descendents = {
            a: Node(node.state.next_state(a)) for a in A
        }

        return node

    def __init__(self, state, A=None):
        self.A = list(state.available_actions()) if A is None else A
        self.state: 'Hex' = state
        # { a: (N(s, a), W(s, a)) }
        self.statistics = {}
        # { a: child node }
        self.descendents = {}

    def is_leaf(self):
        return not self.descendents

    def expand(self, root: 'MCTS'):
        priors = root.default_policy.desirabilities(self.state, self.A)
        self.statistics = {a: Edge(W=0, N=0, P=p)
                           for a, p in zip(self.A, priors)}
        self.descendents = {a: Node(self.state.next_state(a)) for a in self.A}

    def _co_expand(self, root: 'MCTS'):
        A = self.A
        priors = (yield self.state).desirabilities(A)

        self.statistics = {a: Edge(W=0, N=0, P=p) for a, p in zip(A, priors)}
        self.descendents = {a: Node(self.state.next_state(a)) for a in A}

    def evaluate(self, root: 'MCTS') -> float:
        state = self.state
        # We assume that we only get a reward at the end of the game.
        while not state.is_final():
            action = root.default_policy.policy(state)
            state = state.next_state(action)

        winner = state.winner()
        # Tie
        if winner is None:
            return 0
        # Current player won
        elif winner == self.state.current_player:
            return 1
        # Opponent won
        else:
            return -1

    def _co_evaluate(self, root: 'MCTS') -> float:
        state = self.state
        # We assume that we only get a reward at the end of the game.
        while not state.is_final():
            action = (yield state).action(state)
            # action = root.default_policy.policy(state)
            state = state.next_state(action)

        winner = state.winner()
        # Tie
        if winner is None:
            return 0
        # Current player won
        elif winner == self.state.current_player:
            return 1
        # Opponent won
        else:
            return -1

    def _repr_dot_(self, dot):
        board = '\n'.join(''.join(map(str, row)) for row in self.state.grid)
        dot.node(str(id(self)), f'Next: {self.state.current_player}\n{board}')
        for a, (W, N, P) in self.statistics.items():
            other = self.descendents[a]
            if not other.is_leaf():
                dot.edge(str(id(self)), str(id(other)),
                         label=f'W = {W}, N = {N}, P ={P}')
                other._repr_dot_(dot)

    def _repr_svg_(self):
        dot = graphviz.Digraph()
        dot.graph_attr['rankdir'] = 'LR'
        self._repr_dot_(dot)
        return dot._repr_svg_()


class MCTS:
    def __init__(self, default_policy: 'BetaHex', leaf_evaluator: 'LeafEvaluator', c: float = 0.01, terminate=lambda i: i >= 500):
        self.default_policy = default_policy
        self.leaf_evaluator = leaf_evaluator
        self.c = c
        self.terminate = terminate

    def tree_policy(self, node: 'Node'):
        A = node.A

        def Q(a):
            (W, N, P) = node.statistics[a]
            Q = 0 if N == 0 else W / N
            return Q

        def U(a):
            (W, N, P) = node.statistics[a]
            return self.c * P * sum(node.statistics[b].N for b in A) ** 0.5 / (1 + N)

        return max(A, key=lambda a: Q(a) + U(a))

    def policy(self, state: Union[State, Node], callback=lambda r: None):
        root = Node(state)
        it = count()
        # Perform rollout games until we're told to stop (e.g. time limit, max # rollouts, ...)
        while not self.terminate(next(it)):
            self.simulate(root)
            callback(root)

        # We will choose the branch that was visited the most
        return max(root.state.available_actions(),
                   key=lambda a: root.statistics[a][1])

    def generate_episodes(self, initial_state: State, concurrents=16):
        # We will exeucte `concurrents` games in parallell
        def root(): return Node.with_dirichlet(
            initial_state, alpha=0.3, epsilon=0.25, root=self)
        # The games that are awaiting leaf evaluation
        queue = collections.deque(
            [(None, self._co_episode(root())) for _ in range(concurrents)]
        )

        while True:
            batch = []
            # print('hi?')
            for i in range(len(queue)):
                (arg, coroutine) = queue[i]

                try:
                    response = coroutine.send(arg)
                    assert isinstance(response, Hex)
                    queue[i] = (response, coroutine)
                    batch.append(i)
                except StopIteration as stop:
                    yield stop.value
                    queue[i] = (None, self._co_episode(root()))

            outputs = self.default_policy.batch([queue[i][0] for i in batch])
            # actions = self.default_policy.batch_policy(
            #    [queue[i][0] for i in batch])

            for i, o in zip(batch, outputs):
                queue[i] = (o, queue[i][1])
            #trees = [root for _ in range(concurrents)]

    def _co_episode(self, root):
        samples = []
        while not root.state.is_final():
            it = count()
            # Perform simulations until we're told to stop (e.g. time limit, max # simulations, ...)
            while not self.terminate(next(it)):
                yield from self._co_simulate(root)

            # Create the distribution of visits D and add it as a sample
            N = sum(s.N for s in root.statistics.values())
            D = [(a, s.N / N) for a, s in root.statistics.items()]

            samples.append((root.state, D))
            #samples.append((root.state, D))
            # Choose actions in proportion to the root visit counts
            A = root.A
            w = [root.statistics[a].N for a in A]
            action, = choices(A, w, k=1)
            root = root.descendents[action]

        # Since we have reached a final state, we know now the outcome. We combine
        # this with the samples collected to enable us to train both the value function
        # and the policy function
        z = yield from root._co_evaluate(self)
        post_samples = []
        for (s, D) in reversed(samples):
            post_samples.append((s, D, z))
            z = -z
        return post_samples

    def _co_simulate(self, root):
        path = []
        current = root
        while not current.is_leaf():
            action = self.tree_policy(current)
            path.append((current, action))
            current = current.descendents[action]

        # We'll perform an evaluation of the leaf node through e.g. rollout,
        # and backpropagate the result
        value = - (yield from current._co_evaluate(self))
        for (node, a) in reversed(path):
            (W, N, P) = node.statistics[a]
            node.statistics[a] = Edge(W + value, N + 1, P)
            # To ensure that the value is always seen wrt. the current player
            # (we assume a zero-sum game)
            value *= -1
        # We will expand the leaf node, enabling us to dig further at a later time
        yield from current._co_expand(self)

    def episode(self, initial_state: State, return_full_tree=False, callback=lambda r: None):
        # We can summarize the search as follows:
        # yield    what                                 | note
        #       1. Initialize root node.                | done after an episode has ended (and when first resuming)
        #       2. In-tree-traversal until leaf node.   | must keep track of path
        #   *   3. Rollout                              | must keep track of current state in rollout
        #   *   4. Expand leaf node                     |
        #   *   5. Backtrack                            | yield samples
        #       6. Go to 2
        #
        root = Node.with_dirichlet(
            initial_state, alpha=0.3, epsilon=0.25, root=self)
        initial = root if return_full_tree else None
        samples = []

        while not root.state.is_final():
            it = count()
            # Perform rollout games until we're told to stop (e.g. time limit, max # rollouts, ...)
            while not self.terminate(next(it)):
                self.simulate(root)
                callback(initial)

            # Create the distribution of visits D and add it as a sample
            N = sum(s.N for s in root.statistics.values())
            D = [(a, s.N / N) for a, s in root.statistics.items()]
            samples.append((root.state, D))
            # Choose actions in proportion to the root visit counts
            A = root.A
            w = [root.statistics[a].N for a in A]
            action, = choices(A, w, k=1)
            root = root.descendents[action]

        # Since we have reached a final state, we know now the outcome. We combine
        # this with the samples collected to enable us to train both the value function
        # and the policy function
        #z = root.evaluate(self)
        z = Rollout(None).evaluate(root.state)
        post_samples = []
        for (s, D) in reversed(samples):
            post_samples.append((s, D, z))
            z = -z

        return (post_samples, initial) if return_full_tree else post_samples

    def fit(self, samples):
        self.default_policy.fit(samples)

    def simulate(self, root):
        path = []

        current = root

        while not current.is_leaf():
            action = self.tree_policy(current)
            path.append((current, action))
            current = current.descendents[action]

        # We'll perform an evaluation of the leaf node through e.g. rollout,
        # and backpropagate the result
        value = -self.leaf_evaluator.evaluate(current.state)
        for (node, a) in reversed(path):
            (W, N, P) = node.statistics[a]
            node.statistics[a] = Edge(W + value, N + 1, P)
            # To ensure that the value is always seen wrt. the current player
            # (we assume a zero-sum game)
            value *= -1
        # We will expand the leaf node, enabling us to dig further at a later time
        current.expand(self)

    def rebase(self, action, reply) -> 'Node':
        # TODO: Should probably include a fallback in case the reply is unexpanded?
        node = self.root

        for a in [action, reply]:
            if a in node.descendents:
                node = node.descendents[a]
            else:
                node = Node(node.state.next_state(a))

        return node
