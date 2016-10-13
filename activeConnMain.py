# coding: utf-8

##              Bayesian RNN for Active Connectomics Learning
''' 
     Bayesian neural network that allows the prediction of structual connectivity,
     coupled with pertubation of the system for stronger inferences. 
'''
# Imports --------------------------------------------------------------------------



from optoConn.optoConnSet import optoConn
from scipy.io             import loadmat
from optoConn.tools       import *

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
    batchSize : Number of example per batch. Currently HAS to be 1
                when using classification models.

    detail    : If True, details of the training will be printed
                and saved, which slows down the computations.

    dispStep  : Number of iterations before display training info

    keepProb  : Keeping probability for the dropout regularizer

    learnRate : Training learning rate

    nbIters   : Number of training iterations

    sampRate  : Rate at which variables sampling is done

    v2track   : List of name of variables to track (sample)
                 ~> If set to 0, no sampling will be performed


    COST
    -----------
    sparsW    : Weight of the parameter sparsity constraint.
    lossW     : Weight of the lost parameter



    NETWORK
    ------------
    actfct     : Model's activation function for NN 
    model      : Model name to use. List of models ;
                  ~> __classOptoNN__ 
                  ~> __classOptoRNN__
                  ~> __NAR__      

                 (see graph.py for more information)
    
    nhidGlob * : Number of hidden units in global  dynamic cell

    nhidNetw * : Number of hidden units in network dynamic cell

    nInput   * : Number of inputs units

    nOut     * : Number of output units

    nhidclassi : Number of classification units per layer

    multiLayer : Number of layers 



    DATA
    -------
    cells : List of 2 elements containing idx of cells to classify.

               1st cell is the stimulated cell.
               2nd cell is the decoding cell.

    ctrl  : What data set to use as a control for classification

               Can be either ;
                   ~> noStim     will use stimulation of other cells
                   ~> spont      will use spontaneous activity 

    prepMethod : Data preparation method (#) on whole dataset

                   1: standardization
                   2: normalization ( data / abs(max(data)) )
                   3: normalization and shift all values to positive
                   4: standardization + normalization
                   5: dividing data by baseline (deltaF/F)

    null     : If True, the labels will be shuffled. 

    seqRange : Range of sequence to use for the models.

                For classification models :
                 
                  List of 2 lists, where the 2 lists represent the 
                  range before stimulation and range after stimulation.

                For the activity prediction models:
                 
                  List of 2 elements, the first indicating the number
                  time point fed in the model and the second the number
                  of time point in the future to predict.

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
    _mPath   = '/groups/turaga/home/castonguayp/research/optoConn/' # Main path
    dataset  = 'stblock_30Hz_30ms_25obs_20spar_y97.npy'
    #_mPath   = '/home/phabc/Main/research/janelia/turaga/Shotgun/'
    saveName = 'block.ckpt'

    # algorithm parameters
    detail    = True
    batchSize = 20
    dispStep  = 100
    keepProb  = 0.5   
    learnRate = 0.0005
    nbIters   = 1000
    sampRate  = 0
    v2track   = ['']
    

    #Cost parameters
    sparsW   = .001
    lossW    = 1

    #Network Parameters
    actfct     = tf.nn.relu    
    model      = '__classOptoNN__' 
    nhidclassi = 30
    multiLayer = 3    
    nInput     = 75
    nOut       = 75  

    #For NGC RNN model
    nhidGlob   = 10             
    nhidNetw   = 75           
                           

    # Data parameters
    prepMethod = 1
    null       = False
    ctrl       = 'noStim'
    cells      = [200,200]
    seqRange   = [[-10,-2],[2,10]]
    

    #Graph build and execution --------------------------------------------------------

    #Packing dictionnary

    paramDict = {
            'dataset'   : dataset,    '_mPath'  : _mPath,   'saveName'  : saveName, 
            'learnRate' : learnRate,  'nbIters' : nbIters,  'batchSize' : batchSize, 
            'dispStep'  : dispStep,   'model'   : model,    'actfct'    : actfct,  
            'prepMethod': prepMethod, 'seqRange': seqRange, 'multiLayer': multiLayer,
            'nhidGlob'  : nhidGlob ,  'nhidNetw': nhidNetw, 'nInput'    : nInput,
            'nOut'      : nOut   ,    'sampRate': sampRate, 'v2track'   : v2track,
            'sparsW'    : sparsW,     'lossW'   : lossW,    'nhidclassi': nhidclassi,
            'cells'     : cells,      'ctrl'    : ctrl,     'detail'    : detail,
            'keepProb'  : keepProb,   'null'    : null,
                 }

    #Overwrite any parameters with extra arguments
    if argDict:
        paramDict.update(argDict)

    #Lenght of sequenceskeepProb  = 0.5
    if 'class' in paramDict['model']:
        paramDict['seqLen'] = paramDict['seqRange'][0][1] - paramDict['seqRange'][0][0] + \
                              paramDict['seqRange'][1][1] - paramDict['seqRange'][1][0] 
    else:
        paramDict['seqLen'] = paramDict['seqRange'][0]

    #Loading data 
    dataD = loadDataRand('simul', _mPath, paramDict['dataset'])
    
    if 'class' in paramDict['model']:
        dataD['stimIdx'] = dataD['stimIdx']
    
  
    graph, dataDict, Acc = optoConn(paramDict, dataD, run= run, graph= graph)

    return graph, dataDict, optoConn


def optoV1(argDict = None, run = True, graph= None):
    ''' parameter file for simulation.keepProb  = 0.5
        keepProb  = 0.5
        run     : If running the simulation or not
        argDict : Overwriting certain paramers. Has to be of type dict
         '''

    # General Parameters ---------------------------------------------------------------

    #dataset = 'kmeans0.npy'   #Dataset name
    _mPath   = '/groups/turaga/home/castonguayp/research/optoConn/' # Main path
    dataset  = 'optogenExp.h5'
    #_mPath   = '/home/phabc/Main/research/janelia/turaga/Shotgun/'
    saveName = 'opto.ckpt'

    # algorithm parameters
    detail    = False
    batchSize = 1
    dispStep  = 100     
    learnRate = 0.0005 
    nbIters   = 3000
    sampRate  = 0
    v2track   = ['']
    keepProb  = 10.5

    #Cost parameters
    sparsW   = .0005
    lossW    = 1

    #Network Parameters
    actfct   = tf.nn.relu    
    model    = '__classOptoNN__' 
    nInput   = 348            
    nhidGlob = 10             
    nhidNetw = 348             
    nOut     = 348                        

    nhidclassi = 30
    multiLayer = 3    


    # Data parameters
    prepMethod = 5
    null       = False
    ctrl       = 'noStim'
    cells      = [79,79]
    seqRange   = [[-10,-0],[0,10]]
    

    #Graph build and execution --------------------------------------------------------

    #Packing dictionnary
    paramDict = {
            'dataset'   : dataset,    '_mPath'  : _mPath,   'saveName'  : saveName, 
            'learnRate' : learnRate,  'nbIters' : nbIters,  'batchSize' : batchSize, 
            'dispStep'  : dispStep,   'model'   : model,    'actfct'    : actfct,  
            'prepMethod': prepMethod, 'seqRange': seqRange, 'multiLayer': multiLayer,
            'nhidGlob'  : nhidGlob ,  'nhidNetw': nhidNetw, 'nInput'    : nInput,
            'nOut'      : nOut   ,    'sampRate': sampRate, 'v2track'   : v2track,
            'sparsW'    : sparsW,     'lossW'   : lossW,    'nhidclassi': nhidclassi,
            'cells'     : cells,      'ctrl'    : ctrl,     'detail'    : detail,
            'keepProb'  : keepProb,   'null'    : null,
                 }

    #Overwrite any parameters with extra arguments
    if argDict:
        paramDict.update(argDict)

    #Lenght of sequences
    if 'class' in paramDict['model']:
        paramDict['seqLen'] = paramDict['seqRange'][0][1] - paramDict['seqRange'][0][0] + \
                              paramDict['seqRange'][1][1] - paramDict['seqRange'][1][0] 
    else:
        paramDict['seqLen'] = paramDict['seqRange'][0]

    # Loading data 
    #print('Loading Data      ...')
    dataD = loadDataRand('optoV1', _mPath, dataset)
    dataD.update(loadDataSpont('optoV1', _mPath, paramDict['dataset']))

    graph, dataDict, Acc = optoConn(paramDict, dataD, run= run, graph= graph)

    return graph, dataDict, Acc


def loadDataRand(mainName, mPath, dataset):
    ''' Will load the data for different
        main file of random stim condition'''

    if mainName == 'optoV1':
        dsNoList = ['02','05','06','07']
        #dsNoList =['08','09','10'] #list of datasets to load

        #Loading data (baseline removed)
        dPath = mPath + 'data/' + dataset  # Dataset path
        dat   = h5py.File(dPath)

        #Taking only elements related to dataset selected and renaming
        dat = { key: dat[key] for No in dsNoList for key in dat if No in key}

        #Number of frames per dataset (have to be identical shape)
        if '10' in dsNoList:
            T = dat['dataset10'].shape[1]
        else :
            T = 1

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
