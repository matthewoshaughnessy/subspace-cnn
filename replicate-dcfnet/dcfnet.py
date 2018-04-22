##################################################################
### Replicate results from Qiu et al. 2018, DCFNet (arXiv) ########
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
nEpochs = 2
outputName = sys.argv[1]
outputFile = outputName + ".txt"
outputMat = outputName + ".mat"
subspaceProject = False
if len(sys.argv) > 2 and (sys.argv[2].lower() == 'true'):
    subspaceProject = True
noisyData = False
if len(sys.argv) > 3 and (sys.argv[3].lower() == 'true'):
    noisyData = True

lr_def = 0.005
if len(sys.argv) > 4:
    lr_def = float(sys.argv[4])
print("Default learning rate is: ",lr_def)
momentum_def = 0.9
if len(sys.argv) > 5:
    momentum_def = float(sys.argv[5])
print("Default momentum is: ", momentum_def)
lr_decay = 0.5
if len(sys.argv) > 6:
    lr_decay = float(sys.argv[6])
print("Default learning rate decay is: ",lr_decay)
batch_size = 128
if len(sys.argv) > 7:
    batch_size = int(sys.argv[7])

### helper functions #############################################
def unpickle(file):
   import pickle
   with open(file, 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
   return dict

def printlog(text,filename):
    print(text)
    print(text, file=open(filename,'a'))

### initialization ###############################################
# print selected options
printlog("Pytorch version:", outputFile)
printlog(torch.__version__, outputFile)
printlog("Output name: %s" % (outputName), outputFile)
printlog('Outputting debug data to: %s' % (outputFile), outputFile)
printlog('Outputting matlab data to: %s' % (outputMat), outputFile)
if subspaceProject:
    printlog('Subspace projection ON', outputFile)
else:
    printlog('Subspace projection OFF', outputFile)
if noisyData:
    printlog('Noisy test set ON', outputFile)
else:
    printlog('Noisy test set OFF', outputFile)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# get training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

# get (potentially noisy) test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
if noisyData:
    test_batch_noisy = unpickle('./data/cifar-10-batches-py/test_batch_20dB')
    features = np.reshape(test_batch_noisy,(10000,3,32,32)).astype('uint8')
    features = np.transpose(features,(0,2,3,1))
    testset.test_data = features

testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
nClasses = len(classes)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Define the CNN ###############################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # define network
        self.conv1 = nn.Conv2d(3, 64, (5,5), padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, (5,5), padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, (5,5), padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, nClasses)

        # conv1
        self.F1 = (self.conv1.weight).size()[0]
        self.H1 = (self.conv1.weight).size()[2]
        self.W1 = (self.conv1.weight).size()[3]
        self.dim1 = np.int(0.5*self.H1*self.W1)
        self.basis_indices1 = gen_basis_indices(self.F1,self.H1,self.W1,self.dim1)

        # conv2
        self.F2 = (self.conv2.weight).size()[0]
        self.H2 = (self.conv2.weight).size()[2]
        self.W2 = (self.conv2.weight).size()[3]
        self.dim2 = np.int(0.5*self.H2*self.W2)
        self.basis_indices2 = gen_basis_indices(self.F2,self.H2,self.W2,self.dim2)

        # conv3
        self.F3 = (self.conv3.weight).size()[0]
        self.H3 = (self.conv3.weight).size()[2]
        self.W3 = (self.conv3.weight).size()[3]
        self.dim3 = np.int(0.5*self.H3*self.W3)
        self.basis_indices3 = gen_basis_indices(self.F3,self.H3,self.W3,self.dim3)

        # basis
        self.basis1 = scipy.fftpack.dct(np.eye(self.H1*self.W1),norm='ortho')
        self.basis2 = scipy.fftpack.dct(np.eye(self.H2*self.W2),norm='ortho')
        self.basis3 = scipy.fftpack.dct(np.eye(self.H3*self.W3),norm='ortho')

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

net = Net()

if torch.cuda.is_available():
    net = net.cuda()

### Define a Loss function and optimizer ################################

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr_def, momentum=momentum_def)
#optimizer = torch.optim.Adam(net.parameters(), lr=2e-6)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

#### Train the network #################################################

# debug data
loss_history = np.zeros((len(trainloader),nEpochs))
testaccuracy_history = np.zeros((nEpochs,1))
time_history = np.zeros((nEpochs+1,1))
time_history[0] = time.time();

# train
for epoch in range(nEpochs):  # loop over the dataset multiple times

    if torch.cuda.is_available():
        net.cuda()

    running_loss = 0.0
    #scheduler.step()

    if epoch > 0 and epoch % 10 == 0:
        optimizer.param_groups[0]['lr'] = lr_decay* optimizer.param_groups[0]['lr']

    printlog( 'Epoch %d: lr = %f' % (epoch,float(optimizer.param_groups[0]['lr'])), outputFile)

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
        if i % 10 == 9:    # print every 10 mini-batches
            printlog('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 10), outputFile)
            running_loss = 0.0

            # project weights
            if subspaceProject:
                if torch.cuda.is_available():
                    # on gpu
                    w1 = net.conv1.weight.data.numpy()
                    w2 = net.conv2.weight.data.numpy()
                    w3 = net.conv2.weight.data.numpy()
                    w1p = (subspace_projection(net.dim1,net.w1,net.basis1,net.basis_indices1))
                    w2p = (subspace_projection(net.dim2,net.w2,net.basis2,net.basis_indices2))
                    w3p = (subspace_projection(net.dim3,net.w3,net.basis3,net.basis_indices3))
                    net.conv1.weight.data = (torch.from_numpy(w1p)).type(torch.FloatTensor)
                    net.conv2.weight.data = (torch.from_numpy(w2p)).type(torch.FloatTensor)
                    net.conv3.weight.data = (torch.from_numpy(w3p)).type(torch.FloatTensor)
                    # on cpu
                    #w1 = net.conv1.weight.data.cpu().numpy()
                    #w2 = net.conv2.weight.data.cpu().numpy()
                    #w1p = (subspace_projection(dim1,w1,basis1,basis_indices1))
                    #w2p = (subspace_projection(dim2,w2,basis2,basis_indices2))
                    #net.conv1.weight.data = (torch.from_numpy(w1p)).type(torch.FloatTensor).cuda()
                    #net.conv2.weight.data = (torch.from_numpy(w2p)).type(torch.FloatTensor).cuda()
                else:
                    w1 = net.conv1.weight.data.numpy()
                    w2 = net.conv2.weight.data.numpy()
                    w3 = net.conv2.weight.data.numpy()
                    w1p = (subspace_projection(dim1,w1,basis1,basis_indices1))
                    w2p = (subspace_projection(dim2,w2,basis2,basis_indices2))
                    w3p = (subspace_projection(dim3,w3,basis3,basis_indices3))
                    net.conv1.weight.data = (torch.from_numpy(w1p)).type(torch.FloatTensor)
                    net.conv2.weight.data = (torch.from_numpy(w2p)).type(torch.FloatTensor)
                    net.conv3.weight.data = (torch.from_numpy(w3p)).type(torch.FloatTensor)

    # record accuracy on test set
    correct = 0.0
    total = 0.0
    for data in testloader:

        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), labels.cuda()
            outputs = net(inputs)
        else:
            inputs, labels = Variable(inputs), labels

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    testaccuracy_history[epoch] = correct / total
    printlog('--> Accuracy after epoch %d: %d %%' % (epoch, 100 * correct / total), outputFile)

printlog('Finished Training', outputFile)

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

printlog('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total), outputFile)

#class_correct = list(0. for i in range(nClasses))
#class_total = list(0. for i in range(nClasses))
#for data in testloader:
#    images, labels = data
#    outputs = net(Variable(images))
#    _, predicted = torch.max(outputs.data, 1)
#    c = (predicted == labels).squeeze()
#    for i in range(4):
#        label = labels[i]
#        class_correct[label] += c[i]
#        class_total[label] += 1

#for i in range(nClasses):
#    printlog('Accuracy of %5s : %2d %%' % (
#        classes[i], 100 * class_correct[i] / class_total[i]), outputFile)

printlog('Saving data to mat file...', outputFile)
sio.savemat(outputMat,{
    'loss_history' : loss_history,
    'testaccuracy_history' : testaccuracy_history,
    'time_history' : time_history,
    'coeff_1' : coeff_fil_1_ch_1,
    'coeff_2' : coeff_fil_1_ch_2,
    'coeff_3' : coeff_fil_1_ch_3})
printlog('done!', outputFile)


