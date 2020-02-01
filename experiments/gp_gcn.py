import argparse

from fast_mibo.evaluations import evaluate_gcn
from fast_mibo.constants import *
from bayes_opt import BayesianOptimization


def optimize_gcn(args):
    def gcn_val(epochs, num_neurons, learning_rate, dropout, weight_decay):
        configuration = dict()
        configuration["epochs"] = int(epochs)
        configuration["num_neurons"] = int(num_neurons)
        configuration["learning_rate"] = 10 ** learning_rate
        configuration["dropout"] = dropout
        configuration["weight_decay"] = 10 ** weight_decay
        return evaluate_gcn(configuration, args)

    pbounds = {"epochs": (GCN_EPOCHS_RANGE[0], GCN_EPOCHS_RANGE[1] + 1),
               "num_neurons": (GCN_NUM_NEURONS_RANGE[0], GCN_NUM_NEURONS_RANGE[1] + 1),
               "learning_rate": (GCN_LEARNING_RATE_RANGE[0], GCN_LEARNING_RATE_RANGE[1]),
               "dropout": (GCN_DROPOUT_RANGE[0], GCN_DROPOUT_RANGE[1]),
               "weight_decay": (GCN_WEIGHT_DECAY_RANGE[0], GCN_WEIGHT_DECAY_RANGE[1])
               }
    optimizer = BayesianOptimization(f=gcn_val, pbounds=pbounds)
    optimizer.maximize(init_points=0, n_iter=args.runcount)
    print("Final gcn result:", optimizer.max)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_path", type=str, default="../data/cora")
    parser.add_argument("--runcount", type=int, default=100)
    parser.add_argument("--metric", type=str, default="accuracy")
    args = parser.parse_args()

    optimize_gcn(args)


if __name__ == "__main__":
    main()
