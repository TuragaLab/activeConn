#Connectivity inference : Classification model

''' 
Will evaluate the classification ability of each neuron
with respect to all the stimulated neurons
'''
import numpy as np
import time 
import datetime

from smsphc            import sendSMS
from activeConn        import activeConnMain as ACM
from activeConn.tools  import *
from joblib            import Parallel, delayed

import tensorflow as tf


#______________________________________________________________________________

nJobs =  45
path  = '/groups/turaga/home/castonguayp/research/activeConn/classiMat/perceptron/V1Classi/'


def allNclassi(N):
	#Classify all neurons

    acc = np.zeros(255)

                 
    argDict = {    
                 'seqRange'   : [[-30,0],[0,30]], 
                 'actfct'     : tf.nn.relu, 
                 'nbIters'    : 1000, 
                 'sparsW'     : .005,   
                 'nhidclassi' : 50,
                 'model'      : '__classOptoPercep__',
                 'multiLayer' : 3,
                 'method'     : 1,
                 'learnRate'  : 0.0005,   
                 'detail'     : False, 
                 'ctrl'       : 'noStim',
                 'cells'      : [N,1]
                }

    G,D,L = ACM.optoV1(argDict, run = False) 
    
    for n in range(1,255):
        argDict.update({ 'cells':[N,n] }) 

        G,D,L = ACM.optoV1(argDict, run = True, graph = G)

        acc[n-1] = G.AccTe
        
    return acc


#Cross-validationww
for cross in range(4,10):
    saveName = 'classiRAND_255_50x3_prcptrRelu_30_0_0_30_cross_'+str(cross)

    print('Saving in '+ path + saveName)

    #Executing in parrallel
    t = time.time()
    accAll = Parallel(n_jobs=nJobs)( delayed(allNclassi)(i) 
		                             for i in np.arange(1,255) )

    #Stacking into a single matrix
    accAll = np.vstack(accAll)
    print('\nTotal time:  ' + str(datetime.timedelta(seconds = time.time()-t)))

    np.save(path+saveName,accAll)

    sendSMS('Cross completed:  ' + saveName + '\nElapsed time:  ' +
             str(datetime.timedelta(seconds = time.time()-t)) + '\n\n.')



