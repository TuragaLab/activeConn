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
	_mPath   : Path containing data, ckpt folder and main files
	dataset  : Dataset name to load
	saveName : Name of checkpoint to be saved
				 ~> Will be saved in '_mPath/checkpoints/saveName' 	
		   	  	 ~> A checkpoint will be saved '/tmp/backup.ckpt'
	ALGORITHM
	-----------
	batchSize : Number of example per batch
	dispStep  : Number of iterations before display training info
	learnRate : Training learning rate
	nbIters   : Number of training iterations
	sampRate  : Rate at which variables sampling is done
	v2track   : List of name of variables to track (sample)
				 ~> If set to 0, no sampling will be performed

	NETWORK
	------------
	actfct     : Model's activation function for NN 
	model      : Model name to use
				  '__NGCmodel__' : RNN( Network & Global cell) 
				  				   + Calcium dynamic 
	
	nhidGlob * : Number of hidden units in global  dynamic cell
	nhidNetw   : Number of hidden units in network dynamic cell
	nInput     : Number of inputs  units
	nOut       : Number of output units
	seqLen     : Timeserie length for RNN 

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

	#Parameters -----------------------------------------------------------------------

	#dataset = 'kmeans0.npy'   #Dataset name
	_mPath   = '/groups/turaga/home/castonguayp/research/activeConn/' # Main path
	dataset  = 'FR_RNN.mat'
	#_mPath   = '/home/phabc/Main/research/janelia/turaga/Shotgun/'
	saveName = 'sim.ckpt'

	# algorithm parameters
	batchSize = 20      
	dispStep  = 100
	learnRate = 0.0001
	nbIters   = 10000   
	sampRate  = 100
	v2track   = ['ng_IH_HH/MultiRNNCell/Cell0/BasicRNNCell/Linear/Matrix',
				 'ng_Gmean', 'ng_Gstd', 'alpha_W']

	#Network Parameters
	actfct   = tf.sigmoid 
	model    = '__NGCmodel__'
	nhidGlob = 20       
	nhidNetw = 400      
	nInput   = 400      
	nOut     = 400      
	seqLen   = 50        

	# Data parameters
	method = 2  
	t2Dist = 1  

	#Graph ----------------------------------------------------------------------------

	#Packing dictionnary
	paramDict = {
			'dataset'  : dataset,   '_mPath'  : _mPath,   'saveName' : saveName, 
			'learnRate': learnRate, 'nbIters' : nbIters,  'batchSize': batchSize, 
			'dispStep' : dispStep,  'model'   : model,    'actfct'   : actfct,  
			'method'   : method,    't2Dist'  : t2Dist,   'seqLen'   : seqLen,
			'nhidGlob' : nhidGlob,  'nhidNetw': nhidNetw, 'nInput'   : nInput,
			'nOut'     : nOut   ,   'sampRate': sampRate, 'v2track'  : v2track
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
	_mPath   = '/groups/turaga/home/castonguayp/research/activeConn/' # Main path
	dataset  = 'optogenExp.h5'
	#_mPath   = '/home/phabc/Main/research/janelia/turaga/Shotgun/'
	saveName = 'opto.ckpt'

	# algorithm parameters
	batchSize = 20      
	dispStep  = 100     
	learnRate = 0.00001 
	nbIters   = 10000
	sampRate  = 0
	v2track   = ['ng_IH_HH/MultiRNNCell/Cell0/BasicRNNCell/Linear/Matrix',
				 'ng_Gmean', 'ng_Gstd', 'alpha_W']   

	#Network Parameters
	actfct   = tf.nn.relu     
	model    = '__NGCmodel__' 
	nInput   = 348             
	nhidGlob = 20             
	nhidNetw = 348             
	nOut     = 348            
	seqLen   = 10        	  

	# Data parameters
	method 	= 3   
	t2Dist 	= 1   
	dsNo    = 2  #Dataset number 

	#Graph build and execution --------------------------------------------------------

	#Packing dictionnary
	#Packing dictionnary
	paramDict = {
			'dataset'  : dataset,   '_mPath'  : _mPath,   'saveName' : saveName, 
			'learnRate': learnRate, 'nbIters' : nbIters,  'batchSize': batchSize, 
			'dispStep' : dispStep,  'model'   : model,    'actfct'   : actfct,  
			'method'   : method,    't2Dist'  : t2Dist,   'seqLen'   : seqLen,
			'nhidGlob' : nhidGlob,  'nhidNetw': nhidNetw, 'nInput'   : nInput,
			'nOut'     : nOut   ,   'sampRate': sampRate, 'v2track'  : v2track
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