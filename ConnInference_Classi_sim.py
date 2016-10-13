#Connectivity inference : Classification model

''' 
Will evaluate the classification ability of each neuron
with respect to all the stimulated neurons
'''
import numpy as np
import time 
import datetime

from smsphc            import sendSMS
from optoConn        import optoConnMain as OCM
from optoConn.tools  import *
from joblib            import Parallel, delayed

import tensorflow as tf


#______________________________________________________________________________

nJobs  = 75 #Number of jobs
N      = 75 #Number of neurons in network
nCross = 10 #Number of cross-validations

path  =  '/groups/turaga/home/castonguayp/research/' + \
         'optoConn/classiMat/perceptron/simulClassi/'
mSaveName = 'stblock_N01_30Hz_30ms_25obs_20spar_prcptrRelu_10_0_0_10'


def allNclassi(target):
	#Classify all neurons

    acc = np.zeros(N)

                 
    argDict = {    
                 'seqRange'   : [[-5, 0],[0,20]], 
                 'actfct'     : tf.nn.relu, 
                 'nbIters'    : 1500, 
                 'keepProb'   : 0.5,
                 'sparsW'     : .000005,   
                 'nhidclassi' : 100,
                 'dataset'    : 'stblock_N01_30Hz_25obs_20spar_Opto.npy',
                 'model'      : '__classOptoNN__',
                 'multiLayer' : 3,
                 'method'     : 3,
                 'learnRate'  : 0.0005,   
                 'detail'     : False, 
                 'batchSize'  : 1,
                 'ctrl'       : 'noStim',
                 'cells'      : [target,1]
                }

    G,D,L = OCM.simul(argDict, run = False) 
    
    #Decoding cells
    for decode in range(1,N+1):
        argDict.update({ 'cells':[target,decode] }) 

        G,D,L = OCM.simul(argDict, run = True, graph = G)

        acc[decode-1] = G.AccTe
        
    return acc

t = time.time()
#Cross-validationww
for cross in range(0,nCross):
    saveName = mSaveName + '_cross_' + str(cross)

    print('Saving in '+ path + saveName)

    #Executing in parralle
    accAll = Parallel(n_jobs=nJobs)( delayed(allNclassi)(i) 
		                             for i in np.arange(1,N+1) )

    #Stacking into a single matrix
    accAll = np.vstack(accAll)
    print('\nTotal time:  ' + str(datetime.timedelta(seconds = time.time()-t)))

    np.save(path+saveName,accAll)

sendSMS('Job completed ;\n'  + mSaveName + '\n\nElapsed time ; ' +
         str(datetime.timedelta(seconds = time.time()-t)))



