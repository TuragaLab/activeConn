#Spearmint experiment

from activeConn import activeConnMain as ACM
from   joblib   import Parallel, delayed
import numpy as np

def main(job_id, argDict):

	argDict = {arg: argDict[arg][0] for arg in argDict}

	floatD  = {arg: np.float32(argDict[arg]) for arg in argDict if type(argDict[arg]) is float}
	intD    = {arg: np.int32(argDict[arg])   for arg in argDict if type(argDict[arg]) is int}

	#argDict.update(floatD)
	argDict.update(intD)

	print("job id: {}, argDict: {}".format(job_id, argDict))
	nLoops = 100

	accAll = np.zeros(nLoops)
	for i in range(nLoops):
	    G,D,L = ACM.optoV1(floatD)
	    accAll[i] = -G.AccTe

	acc = np.mean(accAll)

	print('  Acc: {:.6} \n'.format(acc)) 
	return acc 