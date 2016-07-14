# coding: utf-8

##              Bayesian RNN for Active Connectomics Learning
''' 
     Bayesian neural network that allows the prediction of structual connectivity,
     coupled with pertubation of the system for stronger inferences. 
'''
# Imports --------------------------------------------------------------------------

from activeConn.activeConnSet import activeConn
from scipy.io                 import loadmat

import tensorflow as tf

# General Parameters ---------------------------------------------------------------

#dataset = 'kmeans0.npy'   #Dataset name
dataset  = 'FR_RNN.mat'
mPath    = '/groups/turaga/home/castonguayp/research/activeConn/' # Main path
#mPath   = '/home/phabc/Main/research/janelia/turaga/Shotgun/'
saveName = 'sim.ckpt'

# algorithm parameters
learnRate = 0.00005 # Learning rate (steps)
nbIters   = 20000   # Number of training iterations
batchSize = 20      # Number of example per batch
dispStep  = 100     # Number of iterations before display

# Network Parameters
model    = '__multirnn_model__'
actfct   = tf.tanh  # Model's activation function
nInput   = 600      # Number of inputs  
seqLen   = 2        # Temporal sequence length 
nhidGlob = 10       # Number of hidden units in global  dynamic cell
nhidNetw = 600      # Number of hidden units in network dynamic cell ~ Might cause problem if not equal to n_input for now
nOut     = 600      # Number of output units

# Data parameters
method = 1  # Data preparation method (0: stand, 1: norm, 2: norm+shift, 3: norm+std)
t2Dist = 1  # Prediction time distance

# Graph build and execution --------------------------------------------------------

#Packing dictionnary
paramDict = {
		'dataset'  : dataset,   'mPath'   : mPath,    'saveName' : saveName, 
		'learnRate': learnRate, 'nbIters' : nbIters,  'batchSize': batchSize, 
		'dispStep' : dispStep,  'model'   : model,    'actfct'   : actfct,  
		'method'   : method,    't2Dist'  : t2Dist,   'seqLen'   : seqLen,
		'nhidGlob' : nhidGlob,  'nhidNetw': nhidNetw, 'nInput'   : nInput,
		'nOut'     : nOut   
	     }


# Loading data 
print('Loading Data      ...')

dPath = mPath + 'data/' + dataset  # Dataset path
data  = loadmat(dPath)['FR']

#Running activeConn ~ Should be called seperatly 
graph, dataDict = activeConn(paramDict, data)
