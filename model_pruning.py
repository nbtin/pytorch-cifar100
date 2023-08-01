import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.utils import prune
import numpy as np

from utils import get_network
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, required=True, help='net type')
# parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
args = parser.parse_args()

model = get_network(args).to("cuda")

# input of the Cifar-100 dataset is 32*32*3
input = torch.randn(1, 3, 32, 32).to("cuda")


def prune_model(model, pruning_rate):
    """
    Prune a pruning_rate of Conv2D connections in the model
    Source: the following code is taken from 
    ----------
    model: torch.nn.Module
    pruning_rate: float between 0.0 and 1.0
    """
    parameters_to_prune = []
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)

    pruning_method = prune.L1Unstructured
    for layer, parameter_name in parameters_to_prune:
        pruning_method.apply(layer, parameter_name, amount=pruning_rate)

pruning_rate = 0.3
prune_model(model, pruning_rate)

torch.save(model, "./checkpoint/pruned_model.pt")