# This file is to evaluate the model resnet152 compare with the pruned_model.pt on the Cifar-100 test set
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os

from models.resnet import resnet152
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Testing')
parser.add_argument('--model', default='./checkpoints/resnet152.pth', type=str, help='model checkpoint path')
parser.add_argument('--pruned-model', default='./checkpoints/pruned_model.pt', type=str, help='pruned model checkpoint path')
# please specify --gpu to be 1 if you want to run it on cpu
# eg: python eval_model.py --model ./resnet152.pth --pruned-model ./pruned_model.pt --gpu -1
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Define the transform to be applied to the images
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# Load the CIFAR-100 test dataset
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Load the resnet152 model and the pruned model
net = resnet152()
net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
net.eval()

# pruned_net = resnet152()
# pruned_net.load_state_dict(torch.load(args.pruned_model))
# pruned_net.eval()

# Load the entire model from the file
pruned_net = torch.load(args.pruned_model, map_location=torch.device('cpu'))

# Get the state_dict from the loaded model
pruned_state_dict = pruned_net.state_dict()

if use_cuda:
    net.cuda()
    pruned_net.cuda()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()

# Test the resnet152 model
net_loss = 0
net_correct = 0
net_total = 0
net_accuracy = []

print('==> Testing model on Cifar100...')
print('------------------RESNET152------------------')


with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        net_loss += loss.item()
        _, predicted = outputs.max(1)
        net_total += targets.size(0)
        net_correct += predicted.eq(targets).sum().item()
        net_accuracy.append(net_correct / net_total)

        progress_bar(batch_idx, len(testloader), 'ResNet152 Test Loss: %.3f | ResNet152 Test Acc: %.3f%% (%d/%d)'
                     % (net_loss / (batch_idx + 1), 100. * net_correct / net_total, net_correct, net_total))


# sort net_accuracy in descending order
net_accuracy.sort(reverse=False)

print('------------------PRUNED RESNET152------------------')

# Test the pruned model
pruned_loss = 0
pruned_correct = 0
pruned_total = 0
pruned_accuracy = []


with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = pruned_net(inputs)
        loss = criterion(outputs, targets)

        pruned_loss += loss.item()
        _, predicted = outputs.max(1)
        pruned_total += targets.size(0)
        pruned_correct += predicted.eq(targets).sum().item()
        pruned_accuracy.append(pruned_correct / pruned_total)

        progress_bar(batch_idx, len(testloader), 'Pruned ResNet152 Test Loss: %.3f | Pruned ResNet152 Test Acc: %.3f%% (%d/%d)'
                     % (pruned_loss / (batch_idx + 1), 100. * pruned_correct / pruned_total, pruned_correct, pruned_total))


pruned_accuracy.sort(reverse=False)

print('------------------RESULTS------------------')

# Print the final results
print('ResNet152 Test Accuracy: %.3f%% (%d/%d)' % (100. * net_correct / net_total, net_correct, net_total))
print('Pruned ResNet152 Test Accuracy: %.3f%% (%d/%d)' % (100. * pruned_correct / pruned_total, pruned_correct, pruned_total))

print('ResNet152 Top-1 Error Rate: %.3f%%' % (1 - net_accuracy[0]))
print('Pruned ResNet152 Top-1 Error Rate: %.3f%%' % (1 - pruned_accuracy[0]))
print('ResNet152 Top-5 Error Rate: %.3f%%' % (1 - net_accuracy[4]))
print('Pruned ResNet152 Top-5 Error Rate: %.3f%%' % (1 - pruned_accuracy[4]))