#Spearmint experiment

from activeConn import activeConnMain as ACM
import numpy as np

def main(job_id, pDict):
  floatD = {arg: np.float32(pDict[arg]) for arg in pDict if type(pDict[arg]) is float}
  intD   = {arg: np.int32(pDict[arg]) for arg in pDict if type(pDict[arg]) is int}

  floatD.update(intD)

  print("job id: {}, pDict: {}".format(job_id, floatD))
  loss = ACM.optoV1(floatD)[2]
  print('  Loss: {:.6} \n'.format(loss)) 
  return loss 

