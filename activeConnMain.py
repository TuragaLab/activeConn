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

import h5py 
import tensorflow as tf

'''
	_______________________________________________________________

								PARAMETERS
	_______________________________________________________________

	Parameters that could make the models fail are not included in 
	the defaults parameters in order to avoid subtle bugs.

	These parameters have a '*' next to them in the list bellow.

	PATH
	----------
	dataset  : Dataset name to load
	_mPath   : Path containing data, ckpt folder and main files
	saveName : Name of checkpoint to be saved
				 + Will be saved in '_mPath/checkpoints/saveName' 	
		   	  	 + A checkpoint will be saved '/tmp/backup.ckpt'
	ALGORITHM
	-----------
	learnRate : Training learning rate
	nbIters   : Number of training iterations
	batchSize : Number of example per batch
	dispStep  : Number of iterations before display training info

	NETWORK
	------------
	model      : Model name to use
				  '__NGCmodel__' : RNN( Network & Global cell) 
				  				   + calcium dynamic 

	actfct     : Model's activation function for NN 
	nInput     : Number of inputs  units
	seqLen     : Timeserie length for RNN 
	nhidGlob * : Number of hidden units in global  dynamic cell
	nhidNetw   : Number of hidden units in network dynamic cell
	nOut       : Number of output units

	DATA
	---------
	method  : Data preparation method
			  0: standardization
			  1: normalization
			  2: normalization and shift to positive 
			  3: standardization + normalization
	t2Dist : Prediction time distance

	_______________________________________________________________


'''

def simul(argDict = None, run = True):
	''' 
	Parameter file for simulation.
	
	_______________________________________________________________

							ARGUMENTS
	_______________________________________________________________

	argDict : Overwriting certain paramers. Has to be of type dict
	run     : If running the simulation or not

	_______________________________________________________________

	'''

	# General Parameters ---------------------------------------------------------------

	#dataset = 'kmeans0.npy'   #Dataset name
	dataset  = 'FR_RNN.mat'
	_mPath   = '/groups/turaga/home/castonguayp/research/activeConn/' # Main path
	#_mPath   = '/home/phabc/Main/research/janelia/turaga/Shotgun/'
	saveName = 'sim.ckpt'

	# algorithm parameters
	learnRate = 0.0001
	nbIters   = 10000   
	batchSize = 20      
	dispStep  = 100     

	#Network Parameters
	model    = '__NGCmodel__'
	actfct   = tf.tanh 
	nInput   = 600      
	seqLen   = 10        
	nhidGlob = 20       
	nhidNetw = 600      
	nOut     = 600      

	# Data parameters
	method = 1  
	t2Dist = 1  

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
	dataset  = 'optogenExp.h5'
	_mPath   = '/groups/turaga/home/castonguayp/research/activeConn/' # Main path
	#_mPath   = '/home/phabc/Main/research/janelia/turaga/Shotgun/'
	saveName = 'opto.ckpt'

	# algorithm parameters
	learnRate = 0.0001 
	nbIters   = 10000   
	batchSize = 20      
	dispStep  = 100     

	#Network Parameters
	model    = '__NGCmodel__' 
	actfct   = tf.nn.relu     
	nInput   = 348             
	seqLen   = 10        	  
	nhidGlob = 20             
	nhidNetw = 348             
	nOut     = 348            

	# Data parameters
	method 	= 2   
	t2Dist 	= 1   
	dsNo    = 2  #Dataset number 

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

	#Loading data
	dPath = _mPath + 'data/' + dataset  # Dataset path
	data  = h5py.File(dPath)['dataset'+str(dsNo)][:]
	

	#Running activeConn ~ Should be called seperatly 
	graph, dataDict = activeConn(paramDict, data, run= run)

	return graph, dataDict