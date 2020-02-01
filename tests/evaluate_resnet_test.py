import os

from fast_mibo.evaluations import evaluate_resnet

if __name__ == '__main__':
    configuration = {"learning_rate": 0.001, "batch_size": 128, "epochs": 2}
    score = evaluate_resnet(configuration,
                            device='cpu',
                            data_path='../data',
                            dataset='cifar10')
    print("Test accuracy:", score)
