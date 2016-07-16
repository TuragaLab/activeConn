# coding: utf-8

##              Bayesian RNN for Active Connectomics Learning
''' 
     Bayesian neural network that allows the prediction of structual connectivity,
     coupled with pertubation of the system for stronger inferences. 
'''
# Imports --------------------------------------------------------------------------

from activeConn.activeConnSet import activeConn
from scipy.io                 import loadmat
from activeConn.tools 		  import *

import tensorflow as tf


def simul(argDict = None, run = True):
	''' parameter file for simulation.
		
		run     : If running the simulation or not
		argDict : Overwriting certain paramers. Has to be of type dict
		 '''

	# General Parameters ---------------------------------------------------------------

	#dataset = 'kmeans0.npy'   #Dataset name
	dataset  = 'FR_RNN.mat'
	_mPath   = '/groups/turaga/home/castonguayp/research/activeConn/' # Main path
	#_mPath   = '/home/phabc/Main/research/janelia/turaga/Shotgun/'
	saveName = 'sim.ckpt'

	# algorithm parameters
	learnRate = 0.0001  # Learning rate (steps)
	nbIters   = 10000   # Number of training iterations
	batchSize = 20      # Number of example per batch
	dispStep  = 100     # Number of iterations before display

	#Network Parameters
	model    = '__NGCmodel__'
	actfct   = tf.tanh  # Model's activation function
	nInput   = 600      # Number of inputs  
	seqLen   = 5        # Temporal sequence length 
	nhidGlob = 20       # Number of hidden units in global  dynamic cell
	nhidNetw = 600      # Number of hidden units in network dynamic cell 
	nOut     = 600      # Number of output units

	# Data parameters
	method = 1  # Data preparation method (0: stand, 1: norm, 2: norm+shift, 3: norm+std)
	t2Dist = 1  # Prediction time distance

	#Graph build and execution --------------------------------------------------------

	#Packing dictionnary
	paramDict = {
			'dataset'  : dataset,   '_mPath'  : _mPath,   'saveName' : saveName, 
			'learnRate': learnRate, 'nbIters' : nbIters,  'batchSize': batchSize, 
			'dispStep' : dispStep,  'model'   : model,    'actfct'   : actfct,  
			'method'   : method,    't2Dist'  : t2Dist,   'seqLen'   : seqLen,
			'nhidGlob' : nhidGlob,  'nhidNetw': nhidNetw, 'nInput'   : nInput,
			'nOut'     : nOut   
		     	 }

	#Overwrite any parameters with extra arguments
	if argDict:
		paramDict.update(argDict)

	# Loading data 
	print('Loading Data      ...')
	dPath = _mPath + 'data/' + dataset  # Dataset path
	data  = loadmat(dPath)['FR']

	#Running activeConn ~ Should be called seperatly 
	graph, dataDict = activeConn(paramDict, data, run= run)

	return graph, dataDict



def optoV1(argDict = None, run = True):
	''' parameter file for simulation.
		
		run     : If running the simulation or not
		argDict : Overwriting certain paramers. Has to be of type dict
		 '''

	# General Parameters ---------------------------------------------------------------

	#dataset = 'kmeans0.npy'   #Dataset name
	dataset  = 'AOS_20150612_L161_Noise_fixed.mat'
	_mPath   = '/groups/turaga/home/castonguayp/research/activeConn/' # Main path
	#_mPath   = '/home/phabc/Main/research/janelia/turaga/Shotgun/'
	saveName = 'opto.ckpt'

	# algorithm parameters
	learnRate = 0.0001 # Learning rate (steps)
	nbIters   = 10000   # Number of training iterations
	batchSize = 20      # Number of example per batch
	dispStep  = 100     # Number of iterations before display

	#Network Parameters
	model    = '__NGCmodel__' # Model to use
	actfct   = tf.nn.relu     # Model's activation function
	nInput   = 348            # Number of inputs  
	seqLen   = 10        	  # Temporal sequence length 
	nhidGlob = 20             # Number of hidden units in global  dynamic cell
	nhidNetw = 348            # Number of hidden units in network dynamic cell 
	nOut     = 348            # Number of output units

	# Data parameters
	method 	   = 2    # Data preparation method (0: stand, 1: norm, 2: norm+shift, 3: norm+std)
	t2Dist 	   = 1    # Prediction time distance
	percentile = 10   # To remove baseline of calcium data
	binsize    = 2000 # Size of bins to remove baseline

	#Graph build and execution --------------------------------------------------------

	#Packing dictionnary
	paramDict = {
			'dataset'  : dataset,   '_mPath'  : _mPath,   'saveName' : saveName, 
			'learnRate': learnRate, 'nbIters' : nbIters,  'batchSize': batchSize, 
			'dispStep' : dispStep,  'model'   : model,    'actfct'   : actfct,  
			'method'   : method,    't2Dist'  : t2Dist,   'seqLen'   : seqLen,
			'nhidGlob' : nhidGlob,  'nhidNetw': nhidNetw, 'nInput'   : nInput,
			'nOut'     : nOut   
		     	 }

	#Overwrite any parameters with extra arguments
	if argDict:
		paramDict.update(argDict)

	# Loading data 
	print('Loading Data      ...')
	field = 'soma_thresh_traces' # Field of interest in dataset
	dsNo  = 2                    # Dataset 
	
	#Loading data
	dPath = _mPath + 'data/' + dataset  # Dataset path
	data  = loadHDF5Dataset(dPath,field,dsNo).value

	#Formatting data
	dataDict = prepare_data(data, seq_len, method = method, t2_dist = t2_dist)

	#Running activeConn ~ Should be called seperatly 
	graph, dataDict = activeConn(paramDict, data, run= run)

	return graph, dataDict