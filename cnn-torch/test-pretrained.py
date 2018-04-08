##################################################################
### Experiment 3: Test subspace constraints on pretrained model ##
##################################################################

import numpy as np
import scipy
import scipy.io as sio
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models
from torch.autograd import Variable

from Projection.basis_gen import *
from Projection.regularization import *

### parameters ###################################################
nEpochs = 30
outputFile = 'experiment1_out.txt'
outputMat = 'experiment1_out.mat'
subspaceProject = False
noisyData = False

### helper functions #############################################
def unpickle(file):
   import pickle
   with open(file, 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
   return dict

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])])

# get training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=4)

# get (potentially noisy) test data
testset = torchvision.datasets.CocoDetection(root='./data', transform=transform)
if noisyData:
    test_batch_noisy = unpickle('./data/cifar-10-batches-py/test_batch_20dB')
    features = np.reshape(test_batch_noisy,(10000,32,32,3),'F').astype('uint8')
    testset.test_data = features
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

nClasses = len(classes)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Define the CNN ###############################################

net = torchvision.models.resnet18(pretrained=True)

if torch.cuda.is_available():
    net = net.cuda()

### Define basis and basis indices for each conv layer ###########

print(net.__dict__.keys())

print(net.conv1)

# conv1
F1 = (net.conv1.weight).size()[0]
H1 = (net.conv1.weight).size()[2]
W1 = (net.conv1.weight).size()[3]
dim1 = np.int(0.5*H1*W1)
basis_indices1 = gen_basis_indices(F1,H1,W1,dim1)

# full basis
basis1 = scipy.fftpack.dct(np.eye(H1*W1),norm='ortho')

### Define a Loss function and optimizer ################################

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#### Train the network #################################################

# debug data
loss_history = np.zeros((len(trainloader),nEpochs))
testaccuracy_history = np.zeros((nEpochs,1))
time_history = np.zeros((nEpochs+1,1))
time_history[0] = time.time();

# train
for epoch in range(nEpochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if torch.cuda.is_available():
            outputs = net(inputs.cuda())
        else:
            outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # save debug data
        loss_history[i,epoch] = loss.data[0]
        time_history[epoch+1] = time.time()

        # print debug data
        running_loss += loss.data[0]
        if i % 500 == 499:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500), file=open(outputFile,"a"))
            running_loss = 0.0

            # project weights
            if subspaceProject:
                if torch.cuda.is_available():
                    w1 = net.conv1.weight.data.cpu().numpy()
                    w1p = (subspace_projection(dim1,w1,basis1,basis_indices1))
                    net.conv1.weight.data = (torch.from_numpy(w1p)).type(torch.FloatTensor).cuda()
                else:
                    w1 = net.conv1.weight.data.numpy()
                    w1p = (subspace_projection(dim1,w1,basis1,basis_indices1))
                    net.conv1.weight.data = (torch.from_numpy(w1p)).type(torch.FloatTensor)

    # record accuracy on test set
    correct = 0.0
    total = 0.0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    testaccuracy_history[epoch] = correct / total
    print('--> Accuracy after epoch %d: %d %%' % (epoch, 100 * correct / total))
    print('--> Accuracy after epoch %d: %d %%' % (epoch, 100 * correct / total), file=open(outputFile,'a'))

print('Finished Training')
print('Finished Training', file=open(outputFile,"a"))

# Verify that the weights lie in the subspace

if torch.cuda.is_available():
    W1 = net.conv1.weight.data.cpu().numpy()
else:
    W1 = net.conv1.weight.data.numpy()
basis = scipy.fftpack.dct(np.eye(25),norm='ortho')

fil_1 = W1[4,:,:,:]
fil_1_ch_1 = fil_1[0,:,:]
fil_1_ch_2 = fil_1[1,:,:]
fil_1_ch_3 = fil_1[2,:,:]

coeff_fil_1_ch_1 = np.dot(basis.T,np.reshape(fil_1_ch_1,25,'F'))
coeff_fil_1_ch_2 = np.dot(basis.T,np.reshape(fil_1_ch_2,25,'F'))
coeff_fil_1_ch_3 = np.dot(basis.T,np.reshape(fil_1_ch_3,25,'F'))

#plt.figure()
#plt.plot(np.abs(coeff_fil_1_ch_1),'*-')
#plt.figure()
#plt.plot(np.abs(coeff_fil_1_ch_2),'*-')
#plt.figure()
#plt.plot(np.abs(coeff_fil_1_ch_3),'*-')

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total), file=open(outputFile,"a"))

class_correct = list(0. for i in range(nClasses))
class_total = list(0. for i in range(nClasses))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(nClasses):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]), file=open(outputFile,"a"))

print('Saving data to mat file...')
print('Saving data to mat file...', file=open(outputFile,'a'))
sio.savemat(outputMat,{
    'loss_history' : loss_history,
    'testaccuracy_history' : testaccuracy_history,
    'time_history' : time_history,
    'class_correct' : class_correct,
    'class_total' : class_total,
    'coeff_1' : coeff_fil_1_ch_1,
    'coeff_2' : coeff_fil_1_ch_2,
    'coeff_3' : coeff_fil_1_ch_3})
print('done!')
print('done!', file=open(outputFile,'a'))


