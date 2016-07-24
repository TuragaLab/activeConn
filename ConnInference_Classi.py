#Connectivity inference : Classification model

''' 
Will evaluate the classification ability of each neuron
with respect to all the stimulated neurons
'''
import numpy as np
import time 
import datetime

from activeConn        import activeConnMain as ACM
from activeConn.tools  import *
from joblib            import Parallel, delayed


#______________________________________________________________________________

nJobs = 85


def allNclassi(N):
	#Classify all neurons

    acc = np.zeros(348)

    argDict = {  'learnRate': 0.0001, 'nbIters':5000, 'nhidclassi':50,
                  'ctrl':'noStim','method':5,  'cells':[N,1], 
                  'detail':False, 'seqRange':[-1,1]  }

    G,D,L = ACM.optoV1(argDict, run = False) 
    
    for n in range(348):
        argDict.update({ 'cells':[N,n] }) 

        G,D,L = ACM.optoV1(argDict, run = True, graph = G)

        acc[n] = G.finalAcc
        
    return acc
    

#Executing in parrallel
t = time.time()
accAll = Parallel(n_jobs=nJobs)( delayed(allNclassi)(i) 
	                                      for i in np.arange(1,255) )

print('\nTotal time:  ' + str(datetime.timedelta(seconds = time.time()-t)))

