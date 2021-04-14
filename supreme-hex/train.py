import argparse
import tensorflow as tf
import numpy as np
import json

from src import actor
from src import games


def load_samples(path):
    with open(path, 'r') as f:
        data = json.load(f)

    shape = data['stateShape']
    samples = data['samples']

    B = []
    X = []
    P = []
    Z = []

    for sample in samples:
        B.append(sample['bitstate'])
        X.append(np.array(sample['state']).reshape(shape))
        P.append(np.array(sample['policy']))
        Z.append(sample['value'])

    return shape, B, X, P, Z


def train(model_path, sample_paths, save_path, size, lr, epochs, optimizer=None, use_new=False):
    shapes = []
    X = []
    P = []
    Z = []

    for path in sample_paths:
        (s, _, x, p, z) = load_samples(path)
        shapes.append(s)
        X.extend(x)
        P.extend(p)
        Z.extend(z)

    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f'mismatching shapes in dataset: {shapes}')

    shape = shapes[0]
    X = tf.stack(X)
    P = tf.stack(P)
    Z = tf.stack(Z)

    loaded = tf.keras.models.load_model(model_path)

    loaded.compile(
        optimizer=optimizer if optimizer is not None else tf.keras.optimizers.SGD(
            lr=lr),
        metrics={'policy': tf.keras.metrics.CategoricalAccuracy(),
                 'value': 'mse'},
        loss_weights={'value': 1.0, 'policy': 1.0},
        loss={'policy': actor.policy_loss, 'value': 'mse'},
    )
    # policy.model.set_weights(loaded.get_weights())
    loaded.fit(X, y={'policy': P, 'value': Z}, epochs=epochs)
    loaded.save(save_path, include_optimizer=False)
    #policy.model.fit(X, y={'policy': P, 'value': Z}, epochs=epochs)
    #policy.model.save(save_path, include_optimizer=False)
    print(f'saved to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)

    args = parser.parse_args()

    train(
        model_path=args.model,
        sample_paths=args.data,
        save_path=args.out,
        size=args.size,
        lr=args.lr,
        epochs=args.epochs,
    )
