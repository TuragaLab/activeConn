import matplotlib.pyplot as plt

import math
import h5py
import collections

import numpy      as np
import tensorflow as tf

pi = math.pi

def batchCreation(inputs,outputs, perm= False, nbIters=100, batchSize=50, seqLen=50):
    ''' 
    ________________________________________________________________________

                                   ARGUMENTS
    ________________________________________________________________________


    inputs    : Input data sequence (time series fed into model)
    outputs   : Output data (label or prediction)
    perm      : Wheter batches will be selected via permutation or random
    nbIters   : Number of batches per batchcreation
    batchSize : number of time series in a batch
    seqLen    : Timeserie length (number of frames)

    ________________________________________________________________________


    '''
    
    nSeq   = inputs.shape[1] - seqLen  # Number of sequences
    nInput = inputs.shape[2]           # Number of inputs

    if perm:
        # Will sample sequences with permutation, which will avoid sampling the same
        # sample multiple times in the same batch
        nSeqB  = nbIters*batchSize #Total number of sequences for all batches

        #Shuffling sequences
        if nSeqB == nSeq:
            perms = np.random.permutation(nSeq) 
            Y1    =  inputs[:,perms,:]
            Y2    = outputs[perms,:]

        elif nSeqB > nSeq:
            nLoop = nSeqB/nSeq #Number of time go through all sequences
            for i in np.arange(np.floor(nLoop)):
                perms = np.random.permutation(nSeq)
                if not i:
                    Y1 =  inputs[:,perms,:]
                    Y2 = outputs[perms,:]
                else:
                    Y1 = np.vstack((Y1, inputs[:,perms,:]))
                    Y2 = np.vstack((Y2,outputs[perms,:]))

            #Residuals
            if nSeqB%nSeq > 0:
                perms = np.random.permutation(nSeq)

                Y1    = np.hstack((Y1, inputs[:,perms[np.arange(nSeqB%nSeq)],:]))
                Y2    = np.vstack((Y2,outputs[perms[np.arange(nSeqB%nSeq)],:]))

        else: 
            perms  = np.random.permutation(nSeq)

            Y1 = inputs[:,perms[np.arange(nSeqB%nSeq)],:]
            Y2 =  outputs[perms[np.arange(nSeqB%nSeq)],:]

    else:

        randidx = np.random.randint(0,nSeq,batchSize*nbIters)

        Y1 = inputs[:,randidx,:]
        Y2 =  outputs[randidx,:]

    return( Y1.reshape([nbIters,batchSize,seqLen,nInput]), 
            Y2.reshape([nbIters,batchSize,nInput]) )


def calcResponse(data, stimFrames, stimOrder, nf = 12, nfb = 1):
    ''' 
    Calcium response when multiple cells are stimulated at the same
    time. 
    ________________________________________________________________________

                                   ARGUMENTS
    ________________________________________________________________________


    data       : Dataset containing optogenetic stimulation
    stimFrames : List of frames where stimulation happens
    stimOrder  : Order in which cells were stimulated
    nf         : Number of frames to keep after stimulation  
    nfb        : Number of frames before stimulation

    ________________________________________________________________________

                                    RETURNS
    ________________________________________________________________________


    trig: Invididual cells repsonse when directly stimulated
    ctrl: Average cells response when other cells are stimulated
    allS: Individual cells response when stimulation of any other cell

    ________________________________________________________________________
 
 

    '''
    nfb = nfb - 1 #correct for python indexing
    tnf = nf+nfb  #total number of frames

    N = data.shape[0]
    nStimCell = len(np.unique(stimOrder))  # Number of cells that are stimulated

    #If multiple neurons at once
    if type(stimOrder) is np.ndarray:
        #Will look at each time each neuron is stimulated directly
        nNeur = stimOrder[0] #number of stimulated neurons at same time

        #Replicating frames 
        repSF = np.tile(stimFrames,[nNeur,1])
        repSF = sorted(repSF)
        repSF = [int(idx[0])for idx in repSF]

        nStim = len(repSF) #Number of stimulations

        # Putting all neurons stimulatead in a single array
        repSIdx = np.hstack(stimOrder.T)

        listStim = np.vstack([np.arange(nStim),repSIdx])
    else:
        #Stimulation cell indexes
        nStim     = len(stimFrames)            # Total number of stimulations
        nReps     = nStim/nStimCell            # Number of time each cell is stimulated

        #Duplicated for repeating stim cycle
        listStim  = np.vstack([np.arange(nStim),np.tile(stimOrder, [1,2])[0]]) 

    cellList  = np.arange(nStimCell) # List of cell idx

    #Ordering stimulation frames based on neuron number 
    ordStim  = listStim[1,:].argsort()

    #Reordering allS based on neuron number
    listAll = np.vstack([np.arange(N*nStim), np.tile(np.arange(N),[1,nStim])[0]])
    ordAll  = listAll[1,:].argsort()

    #Initialization
    trig   = np.zeros([nStim,tnf])     # Average activity when stimulated
    ctrl   = np.zeros([nStimCell,tnf]) # Average activity when other cells are stimulated

    # Taking 1 frame before, 1 frame during and nf frames after the stimulation for each stim
    idx = 0
    for f in ordStim:
        
        frame = stimFrames[f][0] #Stimulation frame
        cell  = listStim[1,f]-1  #Stimulated cell

        if idx == 0: 
            allS = data[:, frame-nfb:frame+nf]
        else:
            allS = np.vstack([allS,data[:, frame-nfb:frame+nf]])
        
        #Index of cells for control except currently stimulated
        mask = np.hstack([cellList!=cell,np.array([False]*(N-nStimCell))])
        
        trig[idx,:]               = data[cell, frame-nfb:frame+nf]
        ctrl[mask[:nStimCell],:] += data[mask, frame-nfb:frame+nf]
        
        idx  += 1
        
    #Taking the mean
    ctrl = ctrl/(nStim-nReps)

    #Difference between direction stimulation and control activity
    diff = trig - np.tile(ctrl,[2,1])

    #Normalizing for each trial
    mTrig = np.tile( np.mean(abs(trig), axis=1), [tnf,1] ).T
    mCtrl = np.tile( np.mean(abs(ctrl), axis=1), [tnf,1] ).T
    mDiff = np.tile( np.mean(abs(diff), axis=1), [tnf,1] ).T
    mAllS = np.tile( np.mean(abs(allS), axis=1), [tnf,1] ).T

    trig = np.divide(trig,mTrig)
    ctrl = np.divide(ctrl,mCtrl)
    diff = np.divide(diff,mDiff) 
    allS = np.divide(allS,mAllS) 

    return trig, diff, allS, ctrl


def calcResponseMulti(data, stimFrames, stimOrder, nf = 12, nfb = 1):
    ''' Calcium response when multiple cells are stimulated at the same
        time. 

    ________________________________________________________________________

                                   ARGUMENTS
    ________________________________________________________________________



    data       : Dataset containing optogenetic stimulation
    stimFrames : List of frames where stimulation happens
    stimOrder  : Order in which cells were stimulated
    nf         : Number of frames to keep after stimulation  
    nfb        : Number of frames before stimulation


    ________________________________________________________________________

                                   RETURNS
    ________________________________________________________________________



    trig: Invididual cells repsonse when directly stimulated
    ctrl: Average cells response when other cells are stimulated
    allS: Individual cells response when stimulation of any other cell


    ________________________________________________________________________
 

    '''
    nfb = nfb - 1 #correct for python indexing
    tnf = nf+nfb  #total number of frames

    N = data.shape[0]
    nStimCell = len(np.unique(stimOrder))  # Number of cells that are stimulated

    #If multiple neurons at once
    if type(stimOrder) is np.ndarray:
        #Will look at each time each neuron is stimulated directly
        nNeur = stimOrder.shape[1] #number of stimulated neurons at same time

        #Replicating frames 
        stimFrames = np.tile(stimFrames,[nNeur,1])
        stimFrames = sorted(stimFrames)
        stimFrames = [int(idx[0])for idx in stimFrames]

        nStim = len(stimFrames) #Number of stimulations
        
        # Putting all neurons stimulatead in a single array
        stimOrder = np.hstack(stimOrder.T)
        
        # List of stimuli
        listStim = np.vstack([np.arange(nStim),stimOrder])

        
        # Number of time each neuron is stimulated
        stimCount  = collections.Counter(stimOrder)
        lstimCount = np.array([stimCount[i] for i in range(1,nStimCell+1)])

    else:
        #Stimulation cell indexes
        nStim     = len(stimFrames)            # Total number of stimulations
        nReps     = nStim/nStimCell            # Number of time each cell is stimulated

        #Duplicated for repeating stim cycle
        listStim  = np.vstack([np.arange(nStim),np.tile(stimOrder, [1,2])[0]]) 


    cellList  = np.arange(nStimCell) # List of cell idx

    #Ordering stimulation frames based on neuron number 
    ordStim  = listStim[1,:].argsort()

    #Reordering allS based on neuron number
    listAll = np.vstack([np.arange(N*nStim), np.tile(np.arange(N),[1,nStim])[0]])
    ordAll  = listAll[1,:].argsort()

    #Initialization
    trig   = np.zeros([nStim,tnf])     # Average activity when stimulated
    ctrl   = np.zeros([nStimCell,tnf]) # Average activity when other cells are stimulated

    # Taking 1 frame before, 1 frame during and nf frames after the stimulation for each stim
    idx = 0
    for f in ordStim:
        
        frame = stimFrames[f] #Stimulation frame
        cell  = listStim[1,f]-1  #Stimulated cell

        if idx == 0: 
            allS = data[:, frame-nfb:frame+nf]
        else:
            allS = np.vstack([allS,data[:, frame-nfb:frame+nf]])
        
        #Index of cells for control except currently stimulated
        mask = np.hstack([cellList!=cell,np.array([False]*(N-nStimCell))])
        
        trig[idx,:]               = data[cell, frame-nfb:frame+nf]
        ctrl[mask[:nStimCell],:] += data[mask, frame-nfb:frame+nf]
        
        idx  += 1
        
    #Taking the mean
    ctrl = (ctrl.T/(nStim-lstimCount)).T

    #Difference between direction stimulation and control activity
    #diff = trig - np.tile(ctrl,[2,1])

    #Normalizing for each trial
    mTrig = np.tile( np.mean(abs(trig), axis=1), [tnf,1] ).T
    mCtrl = np.tile( np.mean(abs(ctrl), axis=1), [tnf,1] ).T
    #mDiff = np.tile( np.mean(abs(diff), axis=1), [tnf,1] ).T
    mAllS = np.tile( np.mean(abs(allS), axis=1), [tnf,1] ).T

    trig = np.divide(trig,mTrig)
    ctrl = np.divide(ctrl,mCtrl)
    #diff = np.divide(diff,mDiff) 
    allS = np.divide(allS,mAllS) 

    return trig, allS, ctrl #diff

def loadHDF5Dataset(path,field,dataset):
    ''' Will return the values associated with the field of 
        the specified dataset. See the README.md file of the 
        dataset for a description of each dataset and field.

        This function is specific to optogenetic data in V1
        from Haüsser lab.

    ________________________________________________________________________

                                   ARGUMENTS
    ________________________________________________________________________


    path    : Path to the dataset
    field   : Name of the field in the dataset to load
    dataset : Number of dataset to load

    ________________________________________________________________________
 

    '''
    
    #To adjust for python indexing
    dataset = dataset - 1
    
    #Open h5py file
    hf = h5py.File(path,'r')
    
    #List of fields and their elements  
    hg = hf['AOS']
    
    #Getting the values of fields inside groups
    vals = [vals for vals in hg.values()]
    
    #List of fields in str
    fields = [f for f in hg.keys()]
    idx    = fields.index(field)
    
    data = hf[vals[idx][dataset][0]]
    
    return data


def meanSTA(data, frames, nstim=8, nf= 241):
    ''' Calculate the calcium triggered average
        

    Assumes repetitive cycles through each stimuli
    in an ordered fashion.
    
    Returns mean STA for each stimuli & neuron
    ________________________________________________________________________

                                   ARGUMENTS
    ________________________________________________________________________


    nstim : Number of stimuli presented
    nf    : Number of frames to save after stimuli
        
    ________________________________________________________________________
 

    '''
    
    N     = data.shape[0]       # Number of neurons
    occur = len(frames)/nstim   # Number of occurences for each stim
    
    #Initialization
    sumSTA = np.zeros([nstim,N,nf]) # Will hold the sum of each elements
    shift  = int(np.floor(nf/4))    # Shifting of window

    #Calculate sum CTA
    stim = 0
    for f in frames:
        sumSTA[stim,:] += data[:,f+1-shift:f+1+nf-shift]
        
        #Increment stim index
        if stim != nstim-1:
            stim += 1
        else:
            stim = 0
       
    return sumSTA/occur  


def prepare_data(data, seqLen, method = 1, t2Dist = 1): 
    ''' Putting the data in the right format for training

    ________________________________________________________________________

                                   ARGUMENTS
    ________________________________________________________________________
 

        seqLen : number of time points 
        method : 0 = No transformation
                 1 = standardization
                 2 = normalization
                 3 = normalization with value shifted to positive
                 4 = standardization + normalization
        t2Dist : Number of timesteps you are tring to predict 

    ________________________________________________________________________

 

    '''

    #Correcting for python index
    t2Dist = t2Dist - 1 

    #Dimensions
    data_num  = data.shape[1]                      # Number of time points
    data_size = data.shape[0]                      # Number of units
    numSeq    = data_num - (seqLen + t2Dist + 1) # Number of sequences
    
    if method == 1:
        # Standardizing
        std  = np.tile( np.std(data, axis = 1),(data_num,1)).T #Matrix of mean for each row 
        mean = np.tile(np.mean(data, axis = 1),(data_num,1)).T #Matrix of std  for each row
        data = np.divide(data-mean,std)

    elif method == 2:
        # Normalizing 
        data = data/np.absolute(data).max(axis=1).reshape(-1, 1)

    elif method == 3:
        # Normalizing with only positive values by shifting values in positive
        data = data+abs(data.min())
        data = data/data.max(axis=1).reshape(-1, 1)

    elif method == 4: 
        # Standardizing
        std  = np.tile( np.std(data, axis = 1),(data_num,1)).T #Matrix of mean for each row 
        mean = np.tile(np.mean(data, axis = 1),(data_num,1)).T #Matrix of std  for each row
        data = np.divide(data-mean,std)

        # Normalizing 
        data = data/np.absolute(data).max(axis=1).reshape(-1, 1)
    
    # Label vectors (T2)
    alOutput  = data[:,seqLen+t2Dist:-1]                 # Take the seqLen+1 vector as output

    #Input vectors (T1)
    alInput   = np.zeros((seqLen, data_size, numSeq))
    for i in range(numSeq):
        alInput[:,:,i] = data[:, i:(i+seqLen)].T  # seqLen * 10x1 vectors before T+1 (exclusive) 
 
    #Putting in float32
    alInput  = np.float32(alInput)
    alOutput = np.float32(alOutput)

    # Five fold cross-validation
    training_num = int(numSeq*4/5) # 80% of the data for training

    #Random permutations of sequences
    perms    = np.random.permutation(numSeq)
    trainIdx = perms[0 : training_num]
    testIdx  = perms[training_num : numSeq]

    #Training set
    trInput  =  alInput[:, :, trainIdx]
    trOutput = alOutput[:,    trainIdx]
    
    #Testing set
    teInput  =  alInput[:, :, testIdx]
    teOutput = alOutput[:,    testIdx]

    #Storing all datasets     
    dataDict = { 'T1'  : trInput.transpose((0,2,1)), 'T2'  : trOutput.transpose(),
                 'Te1' : teInput.transpose((0,2,1)), 'Te2' : teOutput.transpose(),
                 'Fit1': alInput.transpose((0,2,1)), 'Fit2': alOutput.transpose()  }

    return  dataDict


def remBaseline(data, percentile = 10, binsize = 1000):
    ''' 
    Will iterate through a time serie, bin it, calculate the 
    specified percentile and substract this value from the bin.
    This should remove the slower trend not task specific. 

    ________________________________________________________________________

                                   ARGUMENTS
    ________________________________________________________________________
 

    Data       : Has to be size [N x T], where N is number of units 
                        and T is time of serie
    percentile : Percentile value that will be used to substract the bin
    binsize    : Bins size that will be used to correct for drifts

    ________________________________________________________________________
 
    '''

    #Variables
    N    = data.shape[0] #Number of neurons
    T    = data.shape[1] #Number of frames
    half = int(binsize/2)-1 # Number of points on right and left of bin center

    #Data remove baseline
    dataBL = data.copy()                
    
    #Init
    baseline = np.zeros([N,T])

    for t in range(half,T-half):

        #Time points in current bin
        points = range(t-half,t+half+1)

        #Calculating percentile 
        baseline[:,t] = np.percentile(dataBL[:,points], percentile, axis = 1)

    #Filling boundaries baseline
    baseline[:,:half]   = np.tile(baseline[:,  half  ],[half,1]).T 
    baseline[:,T-half:] = np.tile(baseline[:,T-half-1],[half,1]).T

    #Removing baseline
    dataBL = dataBL - baseline

    return dataBL, baseline


def shapeData(_T1, seqLen, nInput):
    '''
    Puts batch data into the following format : seqLen x [batchSize,n_input]
    Taking the inputs
    '''

    _Z1 = tf.identity(_T1)

    # Reshape to prepare input to hidden activation
    _Z1 = tf.reshape(_Z1, [-1, nInput]) # (n_steps*batchSize, n_input)

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _Z1 = tf.split(0, seqLen, _Z1) # n_steps * (batchSize, n_input)
    return _Z1


def stim_nstim_split(data,frameSet):
    '''
        Will split the stimulation part ([-1 frame,stimulation,+1 frame])
        from the non-stimulation part (frames non present in the stimulation section)
        and concatenate the segments.

    '''

    N = data.shape[0]
    
    data_nstim = np.zeros([N,0]) # All the data after and between stimulations ( ]-1,stim,+1[ )
    data_stim  = np.zeros([N,0]) # All the data immediately before (-1)

    nStims = len(frameSet) # Number of frames of stimulations

    for i in range(nStims):
        frames = [f[0] for f in frameSet[i:i+2][:]] # Current and +1 frames

        if i != nStims-1:
            data_nstim = np.hstack( [data_nstim, data[:,frames[0]+2:frames[1]-1]] )
        else:
            #If last frame, will take the next 12 time points
            data_nstim = np.hstack( [data_nstim, data[:,frames[0]+2:frames[0]+15]] )
            
        # Stimulation period ( 3 frames ; -1:stim:+1 )
        data_stim = np.hstack( [data_stim, data[:,frames[0]-1:frames[0]+2]] )
    return data_nstim, data_stim


def varInit(dim, Wname, train = True):
    ''' 
    Will create a tf weight matrix with weight values proportional
    to number of neurons. 

    ________________________________________________________________________

                                   ARGUMENTS
    ________________________________________________________________________
 
    
    dim   : Dimension of the variable to create (can be vector or vector)
    Wname : Name of the variable
    train : If variable is trainable or not.  

    ________________________________________________________________________

    '''

    #Putting in list format
    if type(dim) is int:
        dim = [dim]

    numUnit = sum(dim)  #Total number of units projecting to 

    W = tf.Variable( tf.random_normal( dim, stddev = 1/np.sqrt(numUnit) ), 
                     trainable= train, name= Wname )
    return W


#### - - - - - - - - - - - - - TESTING - - - - - - - - - - - - - ###


def tf_prob_gaussian(x, u, var):
    prob = 1.0/(np.sqrt(2.0*var*pi))*tf.exp(-tf.square(x-u)/(2.0*var))
    return prob


def MyPriorGrad(grad, variable, rho):
    ## add prior term to the existing likelihhod gradient
    fir = rho*tf_prob_gaussian(variable, 0.0, 10.0)  
    sec = (1.0-rho)*tf_prob_gaussian(variable, 0.0, 0.1)
    denom = fir + sec
    num = -(1.0/5.0)*variable*fir - (1/0.1)*variable*sec  
    gradient = grad - 1*num/denom
    gradient = gradient/data_num
    return gradient
