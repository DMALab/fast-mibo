import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from fast_mibo.models import *
from fast_mibo.utils import *


def test(model, test_loader, device):
    model.eval()
    total = 0
    correct = 0
    for (batch_x, batch_y) in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x)

        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    return correct / total


def train_one_epoch(epoch, model, train_loader, device, optimizer):
    model.train()
    for batch_id, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(outputs, batch_y)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        print("Epoch %d, batch_id: %d, loss: %.4f" % (epoch + 1, batch_id + 1, loss.item()))


def evaluate_resnet(configuration, device, data_path, dataset="cifar100"):
    batch_size = configuration["batch_size"]
    learning_rate = configuration["learning_rate"]
    epochs = configuration["epochs"]

    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    num_classes = 0
    if dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_train)
        test_set = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms_test)
        num_classes = 100
    elif dataset == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms_train)
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms_test)
        num_classes = 10
    else:
        raise ValueError("UnSupport dataset!!!")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    model = resnet34(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_one_epoch(epoch, model, train_loader, device, optimizer)


def evaluate_gcn(configuration, args):
    """
    Evaluate the gcn model with pytorch as pygcn.
    :param configuration: the hyperparameter instance of the gcn.
    :return: the performance for optimization.
    """
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.data_path, args.dataset)
    model = GCN(nfeat=features.shape[1],
                nhid=configuration["num_neurons"],
                nclass=labels.max().item() + 1,
                dropout=configuration["dropout"])
    optimizer = optim.Adam(model.parameters(),
                           lr=configuration["learning_rate"],
                           weight_decay=configuration["weight_decay"])
    # start train gcn..
    for epoch in range(configuration["epochs"]):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

    # Get the score of the test/validation score
    model.eval()
    output = model(features, adj)
    acc_test = accuracy(output[idx_test], labels[idx_test])

    return acc_test.item()
