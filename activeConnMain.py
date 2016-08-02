# coding: utf-8

##              Bayesian RNN for Active Connectomics Learning
''' 
     Bayesian neural network that allows the prediction of structual connectivity,
     coupled with pertubation of the system for stronger inferences. 
'''
# Imports --------------------------------------------------------------------------



from activeConn.activeConnSet import activeConn
from scipy.io                 import loadmat
from activeConn.tools         import *

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
    nhidNetw * : Number of hidden units in network dynamic cell
    nInput   * : Number of inputs  units
    nOut     * : Number of output units
    seqLen     : Timeserie length for RNN 

    DATA
    ---------
    method  : Data preparation method
              0: standardization
              1: normalization
              2: normalization and shift to positive 
              3: standardization + normalization
    YDist : Prediction time distance

    _______________________________________________________________


'''

def simul(argDict = None, run = True, graph = None):
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

    ''' parameter file for simulation.
        
        run     : If running the simulation or not
        argDict : Overwriting certain paramers. Has to be of type dict
         '''

    # General Parameters ---------------------------------------------------------------

    #dataset = 'kmeans0.npy'   #Dataset name
    _mPath   = '/groups/turaga/home/castonguayp/research/activeConn/' # Main path
    dataset  = 'stblockOPTO.npy'
    #_mPath   = '/home/phabc/Main/research/janelia/turaga/Shotgun/'
    saveName = 'block.ckpt'

    # algorithm parameters
    detail    = True
    batchSize = 1
    dispStep  = 100     
    learnRate = 0.0005
    nbIters   = 1000
    sampRate  = 0
    v2track   = ['']   

    #Cost parameters
    sparsW   = .005
    lossW    = 1

    #Network Parameters
    actfct   = tf.nn.relu    
    model    = '__classOptoPercep__' 
    nInput   = 1             
    nhidGlob = 10             
    nhidNetw = 348             
    nOut     = 348                        

    nhidclassi = 30
    multiLayer = 3    


    # Data parameters
    method   = 1
    ctrl     = 'noStim'
    cells    = [200,200]
    seqRange = [[-10,-2],[2,10]]
    

    #Graph build and execution --------------------------------------------------------

    #Packing dictionnary

    paramDict = {
            'dataset'  : dataset,   '_mPath'  : _mPath,   'saveName'  : saveName, 
            'learnRate': learnRate, 'nbIters' : nbIters,  'batchSize' : batchSize, 
            'dispStep' : dispStep,  'model'   : model,    'actfct'    : actfct,  
            'method'   : method,    'seqRange': seqRange, 'multiLayer': multiLayer,
            'nhidGlob' : nhidGlob ,  'nhidNetw': nhidNetw, 'nInput'   : nInput,
            'nOut'     : nOut   ,   'sampRate': sampRate, 'v2track'   : v2track,
            'sparsW'   : sparsW,    'lossW'   : lossW,    'nhidclassi': nhidclassi,
            'cells'    : cells,     'ctrl'    : ctrl,     'detail'    : detail,

                 }

    #Overwrite any parameters with extra arguments
    if argDict:
        paramDict.update(argDict)
    #Lenght of sequences
    paramDict['seqLen'] = paramDict['seqRange'][0][1] - paramDict['seqRange'][0][0] + \
                          paramDict['seqRange'][1][1] - paramDict['seqRange'][1][0] 

    #Loading data 
    dataD = loadDataRand('simul', _mPath, dataset)
    dataD['stimIdx'] = dataD['stimIdx']
  
    graph, dataDict, Acc = activeConn(paramDict, dataD, run= run, graph= graph)

    return graph, dataDict, Acc


def optoV1(argDict = None, run = True, graph= None):
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
    detail    = True
    batchSize = 1
    dispStep  = 100     
    learnRate = 0.0005 
    nbIters   = 1000
    sampRate  = 0
    v2track   = ['']   

    #Cost parameters
    sparsW   = .005
    lossW    = 1

    #Network Parameters
    actfct   = tf.nn.relu    
    model    = '__classOptoPercep__' 
    nInput   = 1             
    nhidGlob = 10             
    nhidNetw = 348             
    nOut     = 348                        

    nhidclassi = 30
    multiLayer = 3    


    # Data parameters
    method   = 1
    ctrl     = 'noStim'
    cells    = [200,200]
    seqRange = [[-10,-2],[2,10]]
    

    #Graph build and execution --------------------------------------------------------

    #Packing dictionnary

    paramDict = {
            'dataset'  : dataset,   '_mPath'  : _mPath,   'saveName'  : saveName, 
            'learnRate': learnRate, 'nbIters' : nbIters,  'batchSize' : batchSize, 
            'dispStep' : dispStep,  'model'   : model,    'actfct'    : actfct,  
            'method'   : method,    'seqRange': seqRange, 'multiLayer': multiLayer,
            'nhidGlob' : nhidGlob ,  'nhidNetw': nhidNetw, 'nInput'   : nInput,
            'nOut'     : nOut   ,   'sampRate': sampRate, 'v2track'   : v2track,
            'sparsW'   : sparsW,    'lossW'   : lossW,    'nhidclassi': nhidclassi,
            'cells'    : cells,     'ctrl'    : ctrl,     'detail'    : detail,

                 }

    #Overwrite any parameters with extra arguments
    if argDict:
        paramDict.update(argDict)
    #Lenght of sequences
    paramDict['seqLen'] = paramDict['seqRange'][0][1] - paramDict['seqRange'][0][0] + \
                          paramDict['seqRange'][1][1] - paramDict['seqRange'][1][0] 

    # Loading data 
    #print('Loading Data      ...')
    dataD = loadDataRand('optoV1', _mPath, dataset)
    dataD.update(loadDataSpont('optoV1', _mPath, dataset))

    graph, dataDict, Acc = activeConn(paramDict, dataD, run= run, graph= graph)

    return graph, dataDict, Acc


def loadDataRand(mainName, mPath, dataset):
    ''' Will load the data for different
        main file of random stim condition'''

    if mainName == 'optoV1':

        dsNoList =['08','09','10'] #list of datasets to load

        #Loading data (baseline removed)
        dPath = mPath + 'data/' + dataset  # Dataset path
        dat   = h5py.File(dPath)

        #Taking only elements related to dataset selected and renaming
        dat = { key: dat[key] for No in dsNoList for key in dat if No in key}

        #Number of frames per dataset (have to be identical shape)
        T = dat['dataset10'].shape[1]

        #Initializing data dictionnary
        data = {'dataset':[], 'baseline':[], 'stimFrame':[],'stimIdx':[]}

        frameSet= 0 #To count frame datasets
        for key in sorted(dat):
            if   'base' in key:
                data['baseline'].append(dat[key][:])
            elif 'data' in key:
                data['dataset'].append(dat[key][:])
            elif 'stimF' in key:
                data['stimFrame'].append(dat[key][:]+T*frameSet)
                frameSet +=1 #Correcting for the stacking
            elif 'Idx' in key:
                data['stimIdx'].append(dat[key][:].T)

        #Stacking database
        data = {key: np.hstack(data[key]) for key in data}
        #Transposing stimIdx
        data['stimIdx'] = data['stimIdx'].T

    elif mainName == 'simul':

        dPath = mPath + 'data/' + dataset
        data   = np.load(dPath).all()

    return data


def loadDataSpont(mainName, mPath, dataset):
    ''' Will load the data for different
        main file '''

    if mainName == 'optoV1':

        dsNoList =['07'] #list of datasets to load

        #Loading data (baseline removed)
        dPath = mPath + 'data/' + dataset  # Dataset path
        dat   = h5py.File(dPath)

        #Taking only elements related to dataset selected and renaming
        dat = { key: dat[key] for No in dsNoList for key in dat if No in key}

        #Number of frames per dataset (have to be identical shape)

        #Initializing data dictionnary
        data = {'datasetSpont':[], 'baselineSpont':[] }

        frameSet= 0 #To count frame datasets
        for key in sorted(dat):
            if   'base' in key:
                data['baselineSpont'].append(dat[key][:])
            elif 'data' in key:
                data['datasetSpont'].append(dat[key][:])

        #Stacking database
        data = {key: np.hstack(data[key]) for key in data}

    return data