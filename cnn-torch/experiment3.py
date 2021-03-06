##################################################################
### Experiment 1: visualize weights  #############################
##################################################################

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

### parameters ###################################################
nEpochs = 5
outputName = sys.argv[1]
outputFile = outputName + ".txt"
outputMat = outputName + ".mat"
subspaceProject = False
if len(sys.argv) > 2 and (sys.argv[2].lower() == 'true'):
    subspaceProject = True
noisyData = False
if len(sys.argv) > 3 and (sys.argv[3].lower() == 'true'):
    noisyData = True

### helper functions #############################################
def unpickle(file):
   import pickle
   with open(file, 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
   return dict

# print selected options
print("Output name: %s" % (outputName))
print('Outputting debug data to: %s' % (outputFile))
print('Outputting matlab data to: %s' % (outputMat))
if subspaceProject:
    print('Subspace projection ON')
    print('Subspace projection ON', file=open(outputFile,'a'))
else:
    print('Subspace projection OFF')
    print('Subspace projection OFF', file=open(outputFile,'a'))
if noisyData:
    print('Noisy test set ON')
    print('Noisy test set ON', file=open(outputFile,'a'))
else:
    print('Noisy test set OFF')
    print('Noisy test set OFF', file=open(outputFile,'a'))    

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# get training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# get (potentially noisy) test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
if noisyData:
    test_batch_noisy = unpickle('./data/cifar-10-batches-py/test_batch_20dB')
    features = np.reshape(test_batch_noisy,(10000,3,32,32)).astype('uint8')
    features = np.transpose(features,(0,2,3,1))
    testset.test_data = features
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

convSize1 = 13;
convSize2 = 5;
nClasses = len(classes)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Define the CNN ###############################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, convSize1, stride=1, padding=6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, convSize2)
        self.fc1 = nn.Linear(16 * (convSize2+1) * (convSize2+1), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nClasses)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * (convSize2+1) * (convSize2+1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
if torch.cuda.is_available():
    net = net.cuda()

### Define basis and basis indices for each conv layer ###########

# conv1
F1 = (net.conv1.weight).size()[0]
H1 = (net.conv1.weight).size()[2]
W1 = (net.conv1.weight).size()[3]
dim1 = np.int(0.5*H1*W1)
basis_indices1 = gen_basis_indices(F1,H1,W1,dim1)

# conv2
F2 = (net.conv2.weight).size()[0]
H2 = (net.conv2.weight).size()[2]
W2 = (net.conv2.weight).size()[3]
dim2 = np.int(0.5*H2*W2)
basis_indices2 = gen_basis_indices(F2,H2,W2,dim2)

# full basis
basis1 = scipy.fftpack.dct(np.eye(H1*W1),norm='ortho')
basis2 = scipy.fftpack.dct(np.eye(H2*W2),norm='ortho')

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
                    w2 = net.conv2.weight.data.cpu().numpy()
                    w1p = (subspace_projection(dim1,w1,basis1,basis_indices1))
                    w2p = (subspace_projection(dim2,w2,basis2,basis_indices2))
                    net.conv1.weight.data = (torch.from_numpy(w1p)).type(torch.FloatTensor).cuda()
                    net.conv2.weight.data = (torch.from_numpy(w2p)).type(torch.FloatTensor).cuda()
                else:
                    w1 = net.conv1.weight.data.numpy()
                    w2 = net.conv2.weight.data.numpy()
                    w1p = (subspace_projection(dim1,w1,basis1,basis_indices1))
                    w2p = (subspace_projection(dim2,w2,basis2,basis_indices2))
                    net.conv1.weight.data = (torch.from_numpy(w1p)).type(torch.FloatTensor)
                    net.conv2.weight.data = (torch.from_numpy(w2p)).type(torch.FloatTensor)

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

# save weights
if torch.cuda.is_available():
    W1 = net.conv1.weight.data.cpu().numpy()
    W2 = net.conv2.weight.data.cpu().numpy()
else:
    W1 = net.conv1.weight.data.numpy()
    W2 = net.conv2.weight.data.numpy()

# verify weights lie in the subspace
basis = scipy.fftpack.dct(np.eye(convSize1*convSize1),norm='ortho')
fil_1 = W1[4,:,:,:]
fil_1_ch_1 = fil_1[0,:,:]
fil_1_ch_2 = fil_1[1,:,:]
fil_1_ch_3 = fil_1[2,:,:]
coeff_fil_1_ch_1 = np.dot(basis.T,np.reshape(fil_1_ch_1,convSize1*convSize1,'F'))
coeff_fil_1_ch_2 = np.dot(basis.T,np.reshape(fil_1_ch_2,convSize1*convSize1,'F'))
coeff_fil_1_ch_3 = np.dot(basis.T,np.reshape(fil_1_ch_3,convSize1*convSize1,'F'))

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
    'loss' : loss_history[-1,:],
    'testaccuracy_history' : testaccuracy_history,
    'time_history' : time_history,
    'class_correct' : class_correct,
    'class_total' : class_total,
    'coeff_1' : coeff_fil_1_ch_1,
    'coeff_2' : coeff_fil_1_ch_2,
    'coeff_3' : coeff_fil_1_ch_3,
    'conv1_weights' : W1,
    'conv2_weights' : W2,
    'basis' : basis})
print('done!')
print('done!', file=open(outputFile,'a'))


