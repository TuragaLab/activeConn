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

nJobs =  80
path  =  '/groups/turaga/home/castonguayp/research/' + \
         'activeConn/classiMat/perceptron/simulClassi/'
mSaveName = 'simul_30Hz_y97_prcptrRelu_3_0_0_3'


def allNclassi(N):
	#Classify all neurons

    acc = np.zeros(80)

                 
    argDict = {    
                 'seqRange'   : [[-3, 0],[0,3]], 
                 'actfct'     : tf.nn.relu, 
                 'nbIters'    : 1000, 
                 'sparsW'     : .005,   
                 'nhidclassi' : 30,
                 'dataset'    : 'stblockOPTO_30Hz_y097.npy',
                 'model'      : '__classOptoPercep__',
                 'multiLayer' : 3,
                 'method'     : 1,
                 'learnRate'  : 0.0005,   
                 'detail'     : False, 
                 'ctrl'       : 'noStim',
                 'cells'      : [N,1]
                }

    G,D,L = ACM.simul(argDict, run = False) 
    
    for n in range(1,81):
        argDict.update({ 'cells':[N,n] }) 

        G,D,L = ACM.simul(argDict, run = True, graph = G)

        acc[n-1] = G.AccTe
        
    return acc

t = time.time()
#Cross-validationww
for cross in range(0,20):
    saveName = mSaveName + '_cross_' + str(cross)

    print('Saving in '+ path + saveName)

    #Executing in parrallel
    accAll = Parallel(n_jobs=nJobs)( delayed(allNclassi)(i) 
		                             for i in np.arange(1,81) )

    #Stacking into a single matrix
    accAll = np.vstack(accAll)
    print('\nTotal time:  ' + str(datetime.timedelta(seconds = time.time()-t)))

    np.save(path+saveName,accAll)

sendSMS('Job completed ;\n'  + mSaveName + '\n\nElapsed time ; ' +
         str(datetime.timedelta(seconds = time.time()-t)))



