import argparse


def generate_samples(model_path, out_path, size, samples, simulations, concurrents, leaf_evaluation, encoder):
    import self_play
    import time
    # 2048 seems to be the near the upper limit (based om memory) for V100 and P100 (= 4096 crashes with OOM)
    N = 216 * concurrents if samples is None else samples
    start = time.time()
    self_play.run_save(model_path=model_path, out_path=out_path, size=size, concurrents=concurrents, simulations=simulations,
                       samples=N, leaf_evaluation=leaf_evaluation, encoder=encoder)

    end = time.time()
    throughput = N / (end - start)
    print(f'{concurrents} concurrents')
    print(f'{N} samples to generate')
    print(f'{end - start} s')
    print(f'{throughput} samples/s')
    print(f'saved to {out_path}')
    print(flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--simulations', type=int, default=200)
    parser.add_argument('--c', type=float, default=3.0)
    parser.add_argument('--samples', type=int, required=False)
    parser.add_argument('--concurrents', type=int, required=True)
    parser.add_argument(
        '--evaluation', choices=['value_fn', 'rollout'], default='value_fn')
    parser.add_argument(
        '--encoder', choices=['normalized'], default='normalized')

    args = parser.parse_args()

    generate_samples(
        model_path=args.model,
        out_path=args.out,
        size=args.size,
        samples=args.samples,
        simulations=args.simulations,
        concurrents=args.concurrents,
        leaf_evaluation=args.evaluation,
        encoder=args.encoder,
    )
