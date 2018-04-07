# # Experiment 2
# *Question*: how does generalization error (error on validation set) change as we reduce the amount of unique training examples for both the subspace-constrained and non-subspace-constrained methods?
# *Hypothesis*: Subspace constrained method will have less generalization error than non subspace constrained method as the number of unique training examples decreases
# *Todo*: use subspace dimension from experiment 1

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver
from copy import deepcopy
from IPython.core.debugger import set_trace

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  # returns relative error
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Load the preprocessed CIFAR10 data.
alldata = get_CIFAR10_data()
for k, v in alldata.items():
  print('%s: ' % k, v.shape)



ntrain_total = alldata['X_train'].shape[0]
pct_train_sweep = (1, 0.8, 0.6, 0.4, 0.2)
results_train_accuracy = np.zeros((len(pct_train_sweep),2))
results_test_accuracy  = np.zeros((len(pct_train_sweep),2))
for (i,pct_train) in enumerate(pct_train_sweep):
    # -------------------------
    # --- generate data set ---
    # -------------------------
    ntrain_unique = round(pct_train*ntrain_total)
    ntrain_dupl = ntrain_total - ntrain_unique
    ind_unique = np.arange(0,ntrain_unique)
    ind_dupl = np.random.choice(np.arange(0,ntrain_unique),size=ntrain_dupl)
    ind_train = np.concatenate((ind_unique, ind_dupl))
    print('----- trial %d: %d percent of data (%d unique examples, %d duplicate examples) -----' % (i, pct_train*100, ntrain_unique, ntrain_dupl))
    #print('unique ind:')
    #print(ind_unique)
    #print('duplicate ind:')
    #print(ind_dupl)
    data_abbrev = {
        'X_train': deepcopy(alldata['X_train'])[ind_train,:,:,:],
        'y_train': deepcopy(alldata['y_train'])[ind_train],
        'X_val':   deepcopy(alldata['X_val']),
        'y_val':   deepcopy(alldata['y_val']),
        'X_test':  deepcopy(alldata['X_test']),
        'y_test':  deepcopy(alldata['y_test'])
    }
    # --------------------------------------
    # --- train and report test accuracy ---
    # --------------------------------------
    standardModel  = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=600, reg=0.001)
    subspaceModel  = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=600, reg=0.001)
    standardSolver = Solver(standardModel, data_abbrev,
                            num_epochs=10, batch_size=50,
                            update_rule='adam',
                            optim_config={
                              'learning_rate': 1e-4,
                            },
                            verbose=True, print_every=100)
    subspaceSolver = Solver(subspaceModel, data_abbrev,
                            num_epochs=10, batch_size=50,
                            update_rule='adam',
                            optim_config={
                                'learning_rate': 1e-4,
                            },
                            verbose=True, print_every=100)
    reduced_dim = 24
    print('=== TRAINING STANDARD MODEL FOR TRIAL %d ===' % i)
    standardSolver.train()
    results_train_accuracy[i,0] = standardSolver.train_acc_history[-1]
    results_test_accuracy[i,0] = standardSolver.check_accuracy(alldata['X_test'],alldata['y_test'])
    print('final accuracy for %d percent of data, standard model: %.4f (train), %.4f (test)' % (pct_train*100, results_train_accuracy[i,0], results_test_accuracy[i,1]))
    print('=== TRAINING SUBSPACE MODEL FOR TRIAL %d ===' % i)
    subspaceSolver.train(dim=reduced_dim)
    results_train_accuracy[i,1] = subspaceSolver.train_acc_history[-1]
    results_test_accuracy[i,1] = subspaceSolver.check_accuracy(alldata['X_test'],alldata['y_test'])
    print('final accuracy for %d percent of data, subspace model: %.4f (train), %.4f (test)' % (pct_train*100, results_train_accuracy[i,0], results_test_accuracy[i,1]))



# results from 02/26/2018
#results_train_accuracy = ((0.632, 0.658), (0.692, 0.699), (0.715, 0.701), (0.796, 0.773), (0.919, 0.915))
#results_test_accuracy  = ((0.611, 0.588), (0.587, 0.609), (0.597, 0.584), (0.591, 0.589), (0.565, 0.557))

#print('training accuracy:')
#print(results_train_accuracy)
#print('test accuracy:')
#print(results_test_accuracy)

#plt.subplot(2,1,1)
#set_trace()
#plt.plot(pct_train_sweep,results_train_accuracy[:,0], '-o')
#plt.plot(pct_train_sweep,results_train_accuracy[:,1], '-o')
#plt.legend(['Unconstrained network', 'Subspace constrained network'], loc='upper right')
#plt.xlabel('portion training samples unique')
#plt.ylabel('accuracy')
#plt.title('Training accuracy')
#plt.xlim(0.15,1.05)
#plt.ylim(0,1)

#plt.subplot(2,1,2)
#plt.plot(pct_train_sweep,results_test_accuracy[:,0], '-o')
#plt.plot(pct_train_sweep,results_test_accuracy[:,1], '-o')
#plt.legend(['Unconstrained network', 'Subspace constrained network'], loc='upper left')
#plt.xlabel('portion training samples unique')
#plt.ylabel('accuracy')
#plt.title('Test accuracy')
#plt.xlim(0.15,1.05)
#plt.ylim(0,1)
#plt.show()

