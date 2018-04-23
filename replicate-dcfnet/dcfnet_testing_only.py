import numpy as np
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import scipy
import scipy.fftpack
import scipy.io as sio
from Projection.basis_gen import *
import torch.optim as optim
from Projection.regularization import *
import sys
import time


outputName = sys.argv[1]
outputFile = outputName + ".txt"
outputMat = outputName + ".mat"

filename = './3layerCNN'
if len(sys.argv) > 2:
    filename = sys.argv[2]

noise_std = 0.01
if len(sys.argv) > 3:
    noise_std = sys.argv[3]


def printlog(text,filename):
    print(text)
    print(text, file=open(filename,'a'))


printlog("reading weigts from ",filename, " noise std dev: ", noise_std)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (5,5), padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, (5,5), padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, (5,5), padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x



net = Net()

lr_def = 0.005
momentum_def = 0.9

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr_def, momentum=momentum_def)

net.load_state_dict(torch.load('./3layerCNN'))

print("Weights Loaded!")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=4)

correct_adv = 0
correct = 0
total = 0

count = 0

for data in testloader:
    count = count+1
    print("testing example number ",count)
    inputs, labels = data

    if torch.cuda.is_available():
        inputs, labels = Variable(inputs.cuda(),requires_grad=True), labels.cuda()
        outputs = net(inputs)
    else:
        inputs, labels = Variable(inputs,requires_grad=True), labels
        outputs = net(inputs)
    loss = criterion(outputs, Variable(labels))
    loss.backward()

    # Add perturbation
    epsilon = noise_std
    x_grad   = torch.sign(inputs.grad.data)
   # x_adversarial = torch.clamp(inputs.data + epsilon * x_grad, -1, 1) 
    x_adversarial = inputs.data+epsilon*x_grad
    # Classification after optimization  
    _,y_pred_adversarial = torch.max(net(Variable(x_adversarial)).data,1)

    _, predicted = torch.max(outputs.data, 1)



    total += labels.size(0)
    correct += (predicted == labels).sum()
    correct_adv+= (y_pred_adversarial == labels).sum()

#testaccuracy_history[epoch] = correct / total
printlog('--> Accuracy of this model is: ',(100 * correct / total),filename)
printlog('--> Accuracy of this model on adverserial examples is: ',(100 * correct_adv / total),filename)
