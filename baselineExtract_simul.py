import math
import h5py
import collections
import numpy as np

from scipy.io import loadmat 
from activeConn.tools import *
''' Will remove baseline of all the dataset in h5py file
	and store them in a new h5py file '''

#Parameters
percentile = 10   # To remove baseline of calcium data
binsize    = 3000 # Size of bins to remove baseline

_mPath   = '/groups/turaga/home/castonguayp/research/activeConn/' 

dataset = input('Name of the file containing the dataset to convert?\n')
#dataset  = 'stblockOPTO_W3.mat'
saveName = dataset.rsplit('.',1)[0]

#Removing baseline of dataset
dPath = _mPath + 'data/' + dataset  # Dataset path
sim   = loadmat(dPath)['data']
data  = sim['dataset'][0][0]

print('Removing baseline of dataset ' + dataset)
#data, baseL = remBaseline(data,percentile,binsize)
baseL = np.tile(np.mean(data,axis=1), [data.shape[1],1]).T

#Writing in h5py file name optoExp_0base.h5
print('Storing data')

dat = {
        'dataset'  : data, 
        'baseline' : baseL, 
        'stimFrame': sim['stimFrames'][0][0][0],
        'stimIdx'  : sim['stimIdx'][0][0],
        'W'        : sim['W'][0][0]
       }

np.save(_mPath+'data/'+saveName, dat)