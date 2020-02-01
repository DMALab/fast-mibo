import numpy as np

from fast_mibo.constants import *


def generate_resnet_config():
    pass


def generate_gcn_config():
    config = dict()
    config["epochs"] = np.random.randint(GCN_EPOCHS_RANGE[0], GCN_EPOCHS_RANGE[1] + 1)
    config["num_neurons"] = int(np.random.randint(GCN_NUM_NEURONS_RANGE[0], GCN_NUM_NEURONS_RANGE[1] + 1))
    config["learning_rate"] = 10 ** np.random.uniform(GCN_LEARNING_RATE_RANGE[0], GCN_LEARNING_RATE_RANGE[1])
    config["dropout"] = np.random.uniform(GCN_DROPOUT_RANGE[0], GCN_DROPOUT_RANGE[1])
    config["weight_decay"] = 10 ** np.random.uniform(GCN_WEIGHT_DECAY_RANGE[0], GCN_WEIGHT_DECAY_RANGE[1])
    return config
