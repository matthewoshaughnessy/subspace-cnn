import torch
import torchvision
import torchvision.transforms as transforms
#%matplotlib inline

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

### Define the CNN ###############################################

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net = net.cuda()

### Define basis and basis indices for each conv layer ###########
from Projection.basis_gen import *
import scipy.fftpack

# Conv1
F1 = (net.conv1.weight).size()[0]
H1 = (net.conv1.weight).size()[2]
W1 = (net.conv1.weight).size()[3]
print(F1,H1,W1)

dim1 = np.int(0.5*H1*W1)

basis_indices1 = gen_basis_indices(F1,H1,W1,dim1)

# Conv2
F2 = (net.conv2.weight).size()[0]
H2 = (net.conv2.weight).size()[2]
W2 = (net.conv2.weight).size()[3]
print(F2,H2,W2)

dim2 = np.int(0.5*H2*W2)

basis_indices2 = gen_basis_indices(F2,H2,W2,dim2)

# full basis
basis1 = scipy.fftpack.dct(np.eye(H1*W1),norm='ortho')  
basis2 = scipy.fftpack.dct(np.eye(H2*W2),norm='ortho')    

### Define a Loss function and optimizer ################################

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#### Train the network #################################################

from Projection.regularization import *

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        w1 = net.conv1.weight.data.cpu().numpy()
        w2 = net.conv2.weight.data.cpu().numpy()
        
        w1p  = (subspace_projection(dim1,w1,basis1,basis_indices1))
        w2p  = (subspace_projection(dim2,w2,basis2,basis_indices2))
    
        
        net.conv1.weight.data = (torch.from_numpy(w1p)).type(torch.FloatTensor)
        net.conv2.weight.data = (torch.from_numpy(w2p)).type(torch.FloatTensor)
        
        w1n = net.conv1.weight.data.numpy()
        w2n = net.conv2.weight.data.numpy()
        
        #print(np.linalg.norm(w1 - w1n))
        #print(np.linalg.norm(w2 - w2n))
        
      
        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Verify that the weights lie in the subspace

import scipy
W1 = net.conv1.weight.data.numpy()
print(W1.shape)
basis = scipy.fftpack.dct(np.eye(25),norm='ortho')

fil_1 = W1[4,:,:,:]
fil_1_ch_1 = fil_1[0,:,:]
fil_1_ch_2 = fil_1[1,:,:]
fil_1_ch_3 = fil_1[2,:,:]


print(fil_1_ch_1.shape)

coeff_fil_1_ch_1 = np.dot(basis.T,np.reshape(fil_1_ch_1,25,'F'))
coeff_fil_1_ch_2 = np.dot(basis.T,np.reshape(fil_1_ch_2,25,'F'))
coeff_fil_1_ch_3 = np.dot(basis.T,np.reshape(fil_1_ch_3,25,'F'))

#plt.figure()
#plt.plot(np.abs(coeff_fil_1_ch_1),'*-')
#plt.figure()
#plt.plot(np.abs(coeff_fil_1_ch_2),'*-')
#plt.figure()
#plt.plot(np.abs(coeff_fil_1_ch_3),'*-')

outputs = net(Variable(images))
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

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

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

