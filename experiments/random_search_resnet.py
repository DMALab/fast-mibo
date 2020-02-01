import argparse

from fast_mibo.optimizer import HyperOptimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_count', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='cifar100')
    args = parser.parse_args()

    optimizer = HyperOptimizer()

    optimizer.run()