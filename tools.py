import matplotlib.pyplot as plt

import math
import h5py
import collections

import numpy      as np
import tensorflow as tf

pi = math.pi

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
        stimFrames = np.tile(stimFrames,[1,nNeur]).T
        stimFrames = sorted(stimFrames)
        stimFrames = [int(idx[0])for idx in stimFrames]

        nStim = len(stimFrames) #Number of stimulations
        
        # Putting all neurons stimulatead in a single array
        stimOrder = np.hstack(stimOrder)
        
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
        print(np.arange(nStim).shape)
        print(np.tile(stimOrder, [1,2])[0].shape)
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
        #else:
        #    allS = np.vstack([allS,data[:, frame-nfb:frame+nf]])
        
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



def cellRespMulti(data, stimFrames, stimOrder, nf = 30, nfb = 1):
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



 
def dataPrepClassi(dataDict, ctrl= 'noStim', cells= [217,217], seqRange= [[-2,-1],[0,1]], method = 1):


    ''' Putting the data in the right format for training for 
        classification

    ________________________________________________________________________

                                   ARGUMENTS
    ________________________________________________________________________

        cells[0] : Cell to predict if was stimulated
        cells[1] : Cell to use for predicting if cells[0] was stimulated

        seqRange[0] : Range of time points before stimulation to consider ( [t-u, t-v[ )
        seqRange[1] : Range of time points after  stimulation to consider ( [t+x, t+y[ )

        NEED TO ADJUST THE SPONT STIM BECAUSE OF SEQUENCE BREAKING CAUSED BY NEW SEQRANGE

        method : 0 = No transformation
                 1 = standardization
                 2 = normalization
                 3 = normalization with value shifted to positive
                 4 = standardization + normalization

    ________________________________________________________________________

                                     RETURNS
    ________________________________________________________________________


        dataDict: { Xtr : Training sequences,  Ytr : Traning label1
                    Xte : Testing  sequences,  Yte : Testing label1,
                    Xall: All sequences,       Yall: All labels }

                Format:
                        Xtr, Xte : seqLen x nSequences x nInputs
                        Ytr, Yte : nSquences

    ________________________________________________________________________

    '''

    cells[1] = cells[1] - 1  #Correction for index in python

    F  = dataDict['stimFrame'] #Frame of stimulation
    I  = dataDict['stimIdx']   #Index of neuron stimulated
    B  = dataDict['baseline']
    D  = dataDict['dataset']   #Dataset


    nS = len(F)     #Number of stimulations
    nN = D.shape[0] #Number of units
    sL = seqRange[0][1]-seqRange[0][0]+\
         seqRange[1][1]-seqRange[1][0] #Sequence lenght

    #Preprocessing data (stand or normalization)
    #D =  preProcess(D,  method = method, base = B)
    #DS = preProcess(DS, method = method, base = BS)

    D = D.T #Transposing for final shape

    #Extracting post-stim sequences
    # Will take 'sL' time points around (defined by stimulation frame
    Dstim = np.zeros([sL,nS,nN])
    label = np.zeros(nS)

    for s in range(nS):
        Dstim[:,s,:] = np.vstack([
                       D[ F[s] + seqRange[0][0]: F[s] + seqRange[0][1] ],
                       D[ F[s] + seqRange[1][0]: F[s] + seqRange[1][1] ]  ])
            
        #cells[0] cell label
        if cells[0] in I[s]:
            label[s] = 1
        else:
            label[s] = 0

    Dstim = np.zeros([sL,nS,nN])
    label = np.zeros(nS)

    for s in range(nS):
        Dstim[:,s,:] = np.vstack([
                       D[ F[s] + seqRange[0][0]: F[s] + seqRange[0][1] ],
                       D[ F[s] + seqRange[1][0]: F[s] + seqRange[1][1] ]  ])
            
        #cells[0] cell label
        if cells[0] in I[s]:
            label[s] = 1
        else:
            label[s] = 0

    Dstim = np.float32(Dstim) #For tensorflow

    #Shuffle labels
    perm  = np.random.permutation(len(label))
    label = label[perm]

    #Label counts
    labIdx1 = np.where(label == 1)[0] #Idx of cells[0] stimulation
    labIdx0 = np.where(label == 0)[0] #Idx of non-cells[0] stimulation

    n1 = len(labIdx1) #Number of cells[0] stimulation
    n0 = len(labIdx0) #Number of non-cells[0] stimulation

    #Number of trianing examples
    nTrain1 = int(n1*4/5)
    nTrain0 = int(n0*4/5)


    #Random permutations of sequences
    perms1 = np.random.permutation(n1) #Idx for stimulation
    perms0 = np.random.permutation(n0) #Idx for noStim

    trainIdx1 = perms1[:nTrain1]
    trainIdx0 = perms0[:nTrain0]

    testIdx1 = perms1[nTrain1:]
    testIdx0 = perms0[nTrain0:]

    if ctrl == 'spont':
        #Will use spontaneous activity for no-stim label data
        #print('Using spontaneous data for label 0.\n')

        #Spontaneous datasets
        DS = dataDict['datasetSpont'] #Spontaneous activity dataset
        BS = dataDict['baselineSpont']

        DS = DS.T; BS = BS.T


        #Training inputs
        spontIdxTr = np.random.randint(0,np.shape(DS)[0]-sL,nTrain0)
        SPTr = np.dstack([DS[idx:idx+sL,:] for idx 
                          in spontIdxTr]).transpose(0,2,1)

        #Stacking both label sequences [ Stim, noStim(spont) ]
        trInput = [ Dstim[:,labIdx1[trainIdx1],:], SPTr ]

        #Testing input
        spontIdxTe = np.random.randint(0,np.shape(DS)[0]-sL,n0-nTrain0)
        SPTe = np.dstack([DS[idx:idx+sL,:] for idx 
                          in spontIdxTe]).transpose(0,2,1)

        #Stacking both label sequences [ Stim, noStim(spont) ]
        teInput = [ Dstim[:,labIdx1[testIdx1],:], SPTe ]


    elif ctrl == 'noStim':
        #Training set
        #print('Using random data for label 0.\n')
        trInput = [ Dstim[:,labIdx1[trainIdx1],  :],  #Stim data
                    Dstim[:,labIdx0[trainIdx0], :] ] #No stim data

        #Testing set
        teInput  = [ Dstim[:,labIdx1[testIdx1],  :],
                     Dstim[:,labIdx0[testIdx0], :] ]

    #Taking  only decoding cell
    trInput = [ lab[:,:,cells[1]].T for lab in trInput ] 
    teInput = [ lab[:,:,cells[1]].T for lab in teInput ]

    trInput = [preProcess(lab,  method = method) for lab in trInput]
    teInput = [preProcess(lab,  method = method) for lab in teInput]

    #Stacking label (stim, nostim)
    trLabel = [ label[labIdx1[trainIdx1]], label[labIdx0[trainIdx0]] ]
    teLabel = [ label[labIdx1[testIdx1]],  label[labIdx0[testIdx0]]  ]

    print( 'Stimulated cell : {}\n'.format(cells[0])  +
           'Decoding cell   : {}\n'.format(cells[1]+1)  )

    # One hot  
    # teOutput =  np.zeros([len(teLabel),2])
    # teOutput[np.arange(len(teLabel)),teLabel] = 1  

    dataDict = { 'Xtr' : trInput , 'Ytr' : trLabel,
                 'Xte' : teInput , 'Yte' : teLabel,
                 'Xall': Dstim,    'Yall': label    }

    return dataDict

    

def dataPrepGenerative(data, seqRange= [10,1], method= 1): 
    ''' Putting the data in the right format for training for 
        generative models

    ________________________________________________________________________

                                   ARGUMENTS
    ________________________________________________________________________
 

        seqLen : number of time points 
        method : 0 = No transformation
                 1 = standardization
                 2 = normalization
                 3 = normalization with value shifted to positive
                 4 = standardization + normalization
        YDist : Number of timesteps you are tring to predict 

    ________________________________________________________________________

                                     RETURNS
    ________________________________________________________________________
 

        dataDict: { Xtr   : Training sequences,     Ytr  : Traning label1
                    Xte  : Testing  sequences,     Yte : Testing label1
                    FitX : All training sequences ordered wtr T
                    FitY : Labels of FitX }
                
                Format:
                        Xtr, Xte, FitX : seqLen x nSequences x nInputs
                        Ytr, Yte, FitY : nSquences x nInputs


    ________________________________________________________________________




    '''

    #Correcting for python index
    YDist = YDist - 1 

    data_num  = data.shape[1]                      # Number of time points
    data_size = data.shape[0]                      # Number of units
    numSeq    = data_num - (seqLen + YDist + 1) # Number of sequences   

    #Preprocessing
    data = preProcess(data, method = method)

    # Label vectors (Ytr)
    alOutput  = data[:,seqLen+YDist:-1]                 # Take the seqLen+1 vector as output

    #Input vectors (Xtr)
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
    dataDict = { 'Xtr'  : trInput.transpose((0,2,1)), 'Ytr' : trOutput.transpose(),
                 'Xte'  : teInput.transpose((0,2,1)), 'Yte' : teOutput.transpose(),
                 'FitX' : alInput.transpose((0,2,1)), 'FitY': alOutput.transpose()  }

    return  dataDict




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
    sumSTA  = np.zeros([nstim,N,nf]) # Will hold the sum of each elements
    shift   = int(np.floor(nf/4))    # Shifting of window
    stimDat = []

    #Calculate sum STA
    stim = 0
    for f in frames:
        sumSTA[stim,:] += data[:,f+1-shift:f+1+nf-shift]
        stimDat = np.hstack(stimDat,data[:,f+1-shift:f+1+nf-shift])
        
        #Increment stim index
        if stim != nstim-1:
            stim += 1
        else:
            stim = 0

    meanSTA = sumSTA/occur

    cellResp = np.zeros([8,N])
    for stim in range(8):
        #Integrating overtime
        cellResp[stim,:] = np.sum(meanSTA[stim,:,shift:-shift],axis=1)
        

    return meanSTA, cellResp, stimDat


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

def preProcess(D, method = 1, base = None):
        #Calibrating data 
    nT = D.shape[1] #Number of time point

    #Calibrating data 
    if method == 1:
        # Standardizing
        std  = np.tile( np.std(D, axis = 1),(nT,1)).T #Matrix of mean for each row 
        mean = np.tile(np.mean(D, axis = 1),(nT,1)).T #Matrix of std  for each row
        D = np.divide(D-mean,std)

    elif method == 2:
        # Normalizing 
        D = D/np.absolute(D).max(axis=1).reshape(-1, 1)

    elif method == 3:
        # Normalizing with only positive values by shifting values in positive
        D = D+abs(D.min())
        D = D/D.max(axis=1).reshape(-1, 1)

    elif method == 4: 
        # Standardizing
        std  = np.tile( np.std(D, axis = 1),(nT,1)).T #Matrix of mean for each row 
        mean = np.tile(np.mean(D, axis = 1),(nT,1)).T #Matrix of std  for each row
        D = np.divide(D-mean,std)

        # Normalizing 
        D = D/np.absolute(D).max(axis=1).reshape(-1, 1)

    elif method == 5:
        #Delta f over F 
        D = D/base
        #D = preProcess(D, method = 1, base = None)

    return D


def shapeData(_Xtr, seqLen, nInput):
    '''
    Puts batch data into the following format : seqLen x [batchSize,n_input]
    Taking the inputs
    '''

    _Z1 = tf.identity(_Xtr)

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


def varInit(dim, Wname, ortho = False, train = True, std = None):
    ''' 
    Will create a tf weight matrix with weight values proportional
    to number of neurons. 

    ________________________________________________________________________

                                   ARGUMENTS
    ________________________________________________________________________
 
    
    dim   : Dimension of the variable to create (can be vector or vector)
    Wname : Name of the variable
    train : If variable is trainable or not.
    std   : standart deviation of weights

    ________________________________________________________________________

    '''
    #Putting in list format
    dim = np.int32(dim)
    if type(dim) is int:
        dim = [dim]

    if ortho:
      flat_shape = (dim[0], np.prod(dim[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(dim) #this needs to be corrected to float32

      ortho = tf.constant(std * q[:dim[0], :dim[1]], dtype = tf.float32)

      W = tf.get_variable( Wname, initializer = ortho, 
                                  trainable   = train  )
    else:
        numUnit = sum(dim)  #Total number of units projecting to 

        if not std:
            std = 1/ np.sqrt(numUnit)

        W = tf.get_variable( Wname, initializer = tf.random_normal( dim,
                                                  stddev = std), 
                                    trainable   = train )
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
