# coding: utf-8

# Imports --------------------------------------------------------------------------

import numpy             as np
import matplotlib.pyplot as plt

from os.path           import expanduser 
from scipy.io          import loadmat
from activeConn.graphs import *
from activeConn.tools  import *

import tensorflow as tf


def activeConn(paramDict, data, run = True):
	'''
	Bayesian RNN for Active Connectomics Learning
	       
		 Bayesian neural network that allows the 
		 prediction of causal connectivity coupled with 
		 pertubation of the system for stronger inferences. 
	_______________________________________________________________

							ARGUMENTS
	_______________________________________________________________

	paramDict: Contains all the parameters passed into graph,
		  	   which will be unpacked and added to class attributes
	data     : Data matrix of size (nb_input_units x time)
	run      : Will run the model or not
	_______________________________________________________________

							PARAMETERS
	_______________________________________________________________

	Parameters that could make the models to fail are usually not 
	included in the defaults parameters in order to avoid subtle bugs.

	These parameters have a '*' next to their name in the list bellow.

	PATH
	----------
	dataset  : Dataset name to load
	_mPath   : Main path containing data & ckpt folder and config files
	saveName : Name of checkpoint to be saved
		   	  A checkpoint will also be save as '/tmp/backup.ckpt'

	ALGORITHM
	-----------
	learnRate : Training learning rate
	nbIters   : Number of training iterations
	batchSize : Number of example per batch
	dispStep  : Number of iterations before display of cost information

	NETWORK
	------------
	model      : Model name to use
				   __multirnn_model__ : 
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

	'''

	# Parameters attribution ---------------------------------------------------------
	nInput = data.shape[0] # Number of units in data

	#Default model parameters
	pDict = {  'dataset':'FR_RNN.mat', '_mPath': expanduser("~") + '/.activeConn/',
	      	   'saveName': 'ckpt.ckpt', 'learnRate': 0.0001, 'nbIters':10000,
	           'batchSize': 50, 'dispStep':200, 'model': '__NGCmodel__',
	           'actfct':tf.tanh,'seqLen':10, 'method':1, 't2Dist':1               }       

	#Updatating pDict with input dictionnary
	pDict.update(paramDict)

	#Default for neural network architecture
	if 'nInput'   not in pDict: pDict[ 'nInput'   ] = nInput
	if 'nhidNetw' not in pDict: pDict[ 'nhidNetw' ] = nInput
	if 'nOut'     not in pDict: pDict[ 'nOut'     ] = nInput

	#Verifying if data is consistent with model
	if nInput != pDict['nInput']:
		raise ValueError('Number of input units in the data ({0})'.format(nInput) +
					 ' and number of input units in the model ({0})'.format(pDict['nInput']) +
					 ' need to be equal.')

	# Path
	savepath = pDict['_mPath'] + 'checkpoints/' + pDict['saveName'] # Checkpoint save path


	# Graph build and execution --------------------------------------------------------

	#Formatting data
	dataDict = prepare_data(data, 
						pDict['seqLen'], 
						method = pDict['method'], 
						t2Dist = pDict['t2Dist']
						)

	#Build graph
	print('Building Graph    ...')
	graph = actConnGraph(pDict)

	#Launch Graph
	if run:
		print('Launching Session ...')
		graph.launchGraph( dataDict,
		  				   nbIters  = pDict['nbIters'], 
		  				   dispStep = pDict['dispStep'], 
		                   savepath = savepath )

	return graph, dataDict

