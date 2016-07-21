import time
import datetime

from activeConn.tools import *
from activeConn       import activeConnMain as ACM
from IPython          import display

import numpy                as np
import matplotlib.pyplot    as plt
import tensorflow           as tf
#import matplotlib.animation as anim

from tensorflow.python.ops             import rnn, rnn_cell
from tensorflow.python.ops             import variable_scope        as vs
from tensorflow.python.ops.constant_op import constant  as const


'''
    _________________________________________________________________________

                                 MODELS DESCRIPTION
    _________________________________________________________________________

  

    '__NGCmodel__' : RNN(Network&Globalcell) + Calcium dynamic

                    1. RNN( The global & network cell )
                        
                      Global cell is a low rank RNN with the goal 
                      of capturing global dynamics (fast and slow). 

                      Network cell tries to capture the real dynamic 
                      and connectivity between the units observed in 
                      the data.
                    
                    2. F( Calcium dynamic + 1. ) 
                        
                      Calcium dynamic integrates previous cell 
                      and is also influenced by the previous 
                      inputstep with calcium decay.


    _________________________________________________________________________

                                      FUNCTIONS
    _________________________________________________________________________



    _masking : Adding masks in the tensorflow graph that will 
                  be applied to the designed weights after every
                  variables update.
    
    _VNoise  : Adding stochasticity in all the tensorflow graph 
                  variables after every update for a stotastic 
                  gradient descent. 
    
    launchGraph : Will lauch the training of the model.


    showVars    : Will plot the variables with imshow (matrices)
                  and plot (vectors)

    plotfit     : Will plot the test set and the prediction of 
                  the model.

    _________________________________________________________________________

                                       MASKING
    _________________________________________________________________________



    Masks are used to modulate the parameter space that can be
    explored by the model. They are applied after every variable
    update.

     The first character of a mask has a meaning :
    
                       0 : Mask replace the respective weight
                       1 : Mask is multipled with the respective weight
                       2 : Mask is added to the respective weight
    
     Mask operations will be executed in the same order

     '_M' should be added at the end of the name if the name in the weights 
     dictionnary would be identical otherwise. '_W' can be added to the 
     corresponding name variable instead of '_M' to distinguish them more
     easily.

     ________________________________________________________________________

'''   


class actConnGraph(object):
    # Default graph for Active Connectonic uncovering

    '''
    ________________________________________________________________________

                                     ARGUMENTS
    ________________________________________________________________________
 
    
    featDict: Dictionnary that contains the parameters used to 
              build the graph. To see a list of the parameters,
              see activeConnMain.py file. 

    ________________________________________________________________________

    '''


    def __init__(self, featDict ):

        #Default model parameters
        defaults = {  'learnRate': 0.0001, 'self.nbIters':10000, 'batchSize': 50, 
                      'self.dispStep'  :200, 'model': '__NGCmodel__',
                      'actfct':tf.tanh,'seqLen':10, 'method':1, 'YDist':1   }      

        #Updating default params with featDict
        defaults.update(featDict)

        #Assigining attributes from featDict
        for key, val in defaults.items():
                setattr(self, key, val)

        #Saving dictionnary
        self._pDict = defaults

        #Specifying model
        model = getattr(self, self.model)

        # ---------------------------------------- GRAPH ----------------------------------------- #
        graph = tf.Graph()
        with graph.as_default():    

            #Variable placeholders  
            self._X = tf.placeholder("float", [None, self.seqLen, self.nInput], 
                                     name = 'X') 

            if 'class' in self.model:
                self._Y = tf.placeholder("float", [None, 1], name = 'Y')
            else:
                self._Y = tf.placeholder("float", [None, self.nInput], name = 'Y')
                        
            self._batch = tf.placeholder("int32", [], name = 'batch')                                               

            #Shape data
            _Z1 = shapeData(self._X, self.seqLen, self.nInput)

            #Learning rate decay 
            self.LR = tf.train.exponential_decay(self.learnRate, #LR intial value
                                            self._batch,         #Current batch
                                            500,                 #Decay step
                                            0.98,                #Decay rate
                                            staircase = False)
            
            #Prediction using models
            self._Z2 = model(_Z1)

            #List of all variables 
            self.variables = tf.trainable_variables() 		

            #Variables assigned to each of the variable name 
            self.vnames = {v.name:v for v in self.variables} 

            #Cost function
            cost = self._cost()

            #To test the precision of the network
            self.precision = tf.reduce_mean(tf.pow(self._Z2 - self._Y, 2))
 
            #Backpropagation
            self.optimizer = tf.train.AdamOptimizer( learning_rate = 
                                                     self.LR ).minimize(cost)

            #Adding gaussian noise to variables updates
            #self.V_add_noise = self._VNoise(self.variables) # List of var.assign_add(noise) for all variables

            #Applying masking for restained connectivity
            self.masking = self._masking()

            #Saving graph
            self.saver = tf.train.Saver()

        self.graph = graph


    '''
        ________________________________________________________________________

                                         MODELS
        ________________________________________________________________________
     
    '''

    def __classOpto__(self,_Z1):

        ''' Reccurent neural network with a classifer (logistic) as output layer
            that tries to predicted if there was an otpogenetic stimulation in 
            a neuron j. Input will be time serie of neuron(s) i starting at time t 
            and output will be a binary value, where the label is whether x was 
            stimulated or not at t-z. 


        '''

                #Defining weights
        self.weights = { 
                         'classi_HO_W' : varInit([self.nhidclassi,1], 'classi_HO_W' )
                        }

        self.biases  = { 'classi_HO_B': varInit([1], 'classi_HO_B') } 

        self.masks = { }


        classiCell = rnn_cell.GRUCell(self.nhidclassi)
        initClassi = tf.zeros([self.batchSize, classiCell.state_size])

        #classi
        O, S = rnn.rnn(classiCell, _Z1, dtype = tf.float32) #Output and state

        #classi to output layer
        predCell = tf.matmul(O[-1],self.weights['classi_HO_W'])  + \
                   self.biases['classi_HO_B']

        return predCell

        #Network prediction



    def __NGCmodel__(self,_Z1):
        ''' RNN(Network+Global cells) & Calcium dynamic

            Define weights & masks

                ng_H0_W  : Network&Global hidden -> output (HO)

                alpha    : Decay of data input at t-1
                             0alpha_M: Contrains values between 0 and 1

                ng_IH_HH : Network&Global cell Input  -> Hidden (IH) & Hidden -> Hidden (HH) 
                             
                             1ng_IH_HH: 
                                Mask will be applied so that netw cell receives input from 
                                glob and data, but glob cell only receive data. Furthermore,
                                network cell self-connectivity is prevented by putting the
                                identity to 0. 

                             2ng_IH_HH:
                                Noise is added to this weight matrix for bayesian learning.
                
        '''

        #Total number of hidden units
        nhid = self.nhidNetw + self.nhidGlob

        #Defining weights
        self.weights = { 
                         'ng_H0_W' : varInit([nhid,self.nOut], 'ng_HO_W' ), 
                         'alpha_W' : varInit([self.nInput,1],  'alpha_W' ),
                        }

        #Defining masks
        self.masks = {
                       '1ng_IH_HH': 
                          np.vstack([ 
                                     np.ones([self.nInput, nhid],  dtype='float32'),
                                     np.hstack([ np.ones( [self.nhidNetw]*2,       dtype='float32')   
                                          -np.identity(self.nhidNetw,              dtype='float32'),
                                           np.zeros([self.nhidNetw,self.nhidGlob], dtype='float32') ]),
                                     np.ones([self.nhidGlob,nhid], dtype='float32')
                                    ]),

                       '2ng_IH_HH': tf.random_normal([self.nInput + nhid, nhid],
                                                    0.001) * self.learnRate/2,
                       '0alpha_M' : tf.clip_by_value(self.weights['alpha_W'],0,1)
                      } 

        #Defining biases
        self.biases = { 'ng_H0_B' : varInit(self.nOut, 'ng_H0_B') }  

        #Noise distribution parameters
        ng_Gmean = varInit([1],'ng_Gmean')
        ng_Gstd  = varInit([1],'ng_Gstd' )

        #Network + Global dynamic cell (concatenated)
        ngCell  = rnn_cell.BasicRNNCell(nhid, activation= self.actfct)
        ngCellS = rnn_cell.MultiRNNCell([ngCell])

        #Initialization
        ngO = ngCellS.zero_state(self.batchSize,tf.float32) #Netw+Glob state initialization 
        Z2  = tf.zeros(1)                                   #Model prediction

        #RNN looping through sequence time points
        with tf.variable_scope("ng_IH_HH") as scope:
            for i in range(self.seqLen):

                #Reusing variables for RNN
                if i == 1:
                  scope.reuse_variables()

                #Prediction error for time t
                ZD = _Z1[i] - Z2

                #Network + global cell
                ngO, ngS = ngCellS(ZD, ngO)

                #NG to output cells 
                #ng_Z2 = tf.tanh(tf.matmul(ngO, self.weights['ng_H0_W'] + self.biases['ng_H0_B']))
                ng_Z2 = ngO[:,:self.nhidNetw]

                #Gaussian noise
                gNoise = tf.random_normal([self.batchSize,self.nOut], mean   = ng_Gmean, 
                                                                      stddev = ng_Gstd,  
                                                                      dtype  = 'float32' )
                gNoise = 0

                #Prediction with calcium dynamic
                #Z2 = tf.tanh(tf.matmul(_Z1[i], self.weights['alpha_W']) + ng_Z2 + gNoise)
                Z2 = tf.tanh(ng_Z2 + gNoise)
        
        return Z2


    def __dir_model__(self,_Z1):
        #Building the model following the structure defined under actConnGraph class

        #Total number of hidden units
        nhid = self.nhidNetw + self.nhidGlob

        # Global state initialization
        initNG = tf.zeros((self.batchSize,nhid),          dtype='float32')
        initG  = tf.zeros((self.batchSize,self.nhidGlob), dtype='float32') 
        initN  = tf.zeros((self.batchSize,self.nhidNetw), dtype='float32') 

        #Defining weights
        self.weights = { 
                         'ng_H0_W'   : varInit([nhid,self.nOut], 'ng_HO_W' ), 
                         'alpha_W'   : varInit([self.nInput,1],  'alpha_W' ),
                         'glob_HO_W' : varInit([self.nhidGlob, self.nOut], 'glob_HO_W')
                        }

        #Defining masks
        self.masks = {

                        #Network-Global cell connectivity
                       '1ng_IH_HH': np.vstack([ 
                                     #All input connected
                                     np.ones([self.nInput, nhid],  dtype='float32'), 
                                     
                                     #Network (~I & ~Glob) -> Network
                                     np.hstack([ np.ones( [self.nhidNetw]*2,       dtype='float32')     
                                          -np.identity(self.nhidNetw,              dtype='float32'),
                                           np.zeros([self.nhidNetw,self.nhidGlob], dtype='float32') ]),

                                     #Network&Global -> Global
                                     np.ones([self.nhidGlob,nhid], dtype='float32')
                                                ]),

                       #Network-Global : Adding noise
                       '2ng_IH_HH': tf.random_normal([self.nInput + nhid, nhid],
                                                    0.001) * 0.001,

                       #Calcium decay
                       '0alpha_M' : tf.clip_by_value(self.weights['alpha_W'],0,1),

                       #Network Cell
                      '2netw_IH_HH': tf.random_normal([self.nInput + self.nhidNetw, self.nhidNetw],
                                                    0.001) * 0.001,

                      '1netw_IH_HH':np.vstack([   np.ones( [self.nInput, self.nhidNetw]),
                                                  np.ones( [self.nhidNetw]*2, dtype='float32')     
                                                 -np.identity(self.nhidNetw,  dtype='float32')  ]),

                      '0netw_IH_HH_B': np.zeros(self.nhidNetw),
                      '0glob_IH_HH_B': np.zeros(self.nhidGlob)

                      }

        #Defining biases
        self.biases = {  'ng_H0_B'   : varInit(self.nOut, 'ng_H0_B'  ),
                         'glob_HO_B' : varInit(self.nOut, 'glob_HO_B')  }



        #ngCell   = rnn_cell.BasicRNNCell(nhid,          activation = self.actfct)
        globCell = rnn_cell.BasicRNNCell(self.nhidGlob, activation = self.actfct) #No initialization required
        netwCell = rnn_cell.BasicRNNCell(self.nhidNetw, activation = self.actfct) #No initialization required

        #Global dynamic cell

        #Global dynamic cell output
        globOut, globState = rnn.rnn(globCell, _Z1, initial_state=initG,
                                     dtype=tf.float32, scope = 'glob_IH_HH')
        netwOut, netwState = rnn.rnn(netwCell, _Z1, initial_state=initN,  
                                     dtype=tf.float32, scope = 'netw_IH_HH')
        #ngO, ngS           = rnn.rnn(ngCell, _Z1,   initial_state=initNG, dtype=tf.float32, scope = 'ng_IH_HH')
        
        #ng_Z2    = ngO[-1][:,:self.nhidNetw] 
        
        #Hidden to Output
        glob_Z2 = tf.matmul(globOut[-1], self.weights['glob_HO_W']) # + self.biases['glob_HO_B']

        #Direct netw connectivity
        #netw_Z2 = self.actfct(tf.matmul( _Z1[-1], self.weights['netw_dir_W']) + self.biases['netw_dir_B'])
        #Z2 = tf.matmul(_Z1[-1],self.alpha) + netw_Z2 + glob_Z2

        gNoise = 0

        #Applying calcium filter        
        #Z2 = tf.tanh(tf.matmul(_Z1[-1], self.weights['alpha_W']) + ng_Z2 + gNoise)
        Z2 = tf.matmul(_Z1[-1], self.weights['alpha_W']) + netwOut[-1] + glob_Z2 # + gNoise)
        #Z2 = netwOut[-1] + glob_Z2 

        return Z2


    '''
    ________________________________________________________________________

                                VARIABLE FUNCTIONS
    ________________________________________________________________________

    '''


    def batchCreation(self, inputs, outputs, nbIters=100, perm = True):
        ''' 
        ________________________________________________________________________

                                       ARGUMENTS
        ________________________________________________________________________


        inputs    : Input data sequence (time series fed into model)
        outputs   : Output data (label or prediction)
        perm      : Wheter batches will be selected via permutation or random
        nbIters   : Number of batches per batchcreation

        ________________________________________________________________________

                                     RETURNS
        ________________________________________________________________________


        Y1 : Batches inputs  (nBatches x batchSize x nInput)
        Y2 : Batches outputs (nBatches x batchSize x nInput*)
            *nInput is currently only present in non classifier models

      ________________________________________________________________________



        '''
        if 'class' in self.model: 
            #If model is a classifier
            nSeq = inputs.shape[1]
        else:
            nSeq = inputs.shape[1] - self.seqLen

        if perm:
            # Will sample sequences with permutation, which will avoid sampling the same
            # sample multiple times in the same batch
            nSeqB  = nbIters*self.batchSize #Total number of sequences for all batches


            #Shuffling sequences
            if nSeqB == nSeq:
                perms = np.random.permutation(nSeq) 
                Y1 =  inputs[:,perms,:]
                Y2 = outputs[perms,:]

            elif nSeqB > nSeq:
                nLoop = np.floor(nSeqB/nSeq) #Number of time go through all sequences
                for i in np.arange(nLoop):
                    perms = np.random.permutation(nSeq)
                    if not i:
                        Y1 =  inputs[:,perms,:]
                        Y2 = outputs[perms,...]
                    else:
                        Y1 = np.hstack((Y1, inputs[:,perms,:]))

                        if len(Y2.shape) == 1:#If classi models, Output is 1D
                            Y2 = np.hstack((Y2,outputs[perms,...]))
                        else:
                            Y2 = np.vstack((Y2,outputs[perms,...]))

                #Residuals
                if nSeqB%nSeq > 0:
                    perms = np.random.permutation(nSeq)

                    Y1 = np.hstack((Y1, inputs[:,perms[np.arange(nSeqB%nSeq)],:]))
                    
                    if len(Y2.shape) == 1: #If classi models, Output is 1D
                        Y2 = np.hstack((Y2,outputs[perms[np.arange(nSeqB%nSeq)],...]))
                    else:
                        Y2 = np.vstack((Y2,outputs[perms[np.arange(nSeqB%nSeq)],...]))

            else: 
                perms  = np.random.permutation(nSeq)

                Y1 = inputs[:,perms[np.arange(nSeqB%nSeq)],...]
                Y2 =  outputs[perms[np.arange(nSeqB%nSeq)],...]

        else:

            randidx = np.random.randint(0,nSeq,self.batchSize*nbIters)

            Y1 = inputs[:,randidx,:]
            Y2 =  outputs[randidx,...]

        #Reshapping
        Y1 = Y1.reshape([nbIters,self.batchSize,self.seqLen,self.nInput])
        if len(Y2.shape) == 1: #In classi models, output is 1D
            Y2 = Y2.reshape([nbIters,self.batchSize,1])
        else:
            Y2 = Y2.reshape([nbIters,self.batchSize,self.nInput])

        return Y1, Y2


    def _cost(self):
          ''' Will calculate the costs associated with the loss function '''
          
          #Cost for connectivity in the network cell (H->H)
          
          if 'class' in self.model :
             #If model is a classifier

            #Sparsity regularizer
            sparsC       = [tf.reduce_sum(tf.abs(v)) for v in self.variables]
            self._sparsC = tf.add_n(sparsC)*self.sparsW

            #Cross entropy
            ngml         = tf.nn.sigmoid_cross_entropy_with_logits(self._Z2, self._Y)
            self._ngml   = tf.reduce_sum(ngml)*self.lossW 

            cost = (self._ngml + self._sparsC) / (2*self.batchSize)
                   

          else:
            #sparsC  = tf.add_n([ tf.nn.l2_loss( self.vnames['ng_IH_HH/MultiRNNCell/Cell0/BasicRNNCell/Linear/Matrix:0'][self.nhidNetw:,:] ) ])
            #sparsC = tf.add_n([ tf.reduce_sum( tf.abs( self.vnames[
             #        'ng_IH_HH/BasicRNNCell/Linear/Matrix:0'][self.nhidNetw:,:] ))]) 
            sparsC  = tf.add_n([tf.reduce_sum(tf.abs(v)) for v in self.variables]) 
            #self._sparsC  = tf.add_n([tf.nn.l2_loss(v) for v in self.variables]) 
            #self._sparsC  = tf.reduce_sum(np.float32([0,1]))

            #self._sparsC = (sparsC/self.LR) * (self.learnRate*self.sparsW)

            self._sparsC = (sparsC/self.LR) * (self.learnRate*self.sparsW)

            #Sum of square distance
            self._ngml = tf.reduce_sum(tf.pow(self._Z2 - self._Y, 2))*self.lossW
            
            #Prior
            ngprior = 0

            #Total Cost
            cost = (self._ngml + ngprior + self._sparsC ) / (2*self.batchSize)
            #cost = (self._ngml + ngprior ) / (2*self.batchSize)

          return cost


    def _masking(self):
        ''' Will create the operations to update the weights

        The first character of a mask has a meaning :
        
                           0 : Mask replace the respective weight
                           1 : Mask is added to the respective weight
                           2 : Mask is multipled with the respective weight
        
         Mask operations will be executed in the same order '''

        Vars = self.variables.copy()

        #Names of variables
        vnames = [var.name[:-2] for var in Vars]

        #Will hold variables changes
        tempM =  [var.value() for var in Vars] 

        #Which variables will the masks be applied on
        vidxAll = []     

        # Applying masks
        for m in sorted(self.masks):
            if m[1:] in vnames:
                #If mask is present in variables as it is
                vidx = vnames.index(m[1:]) #Index of variable

                if   m[0] == '0':
                    tempM[vidx] = self.masks[m]
                elif m[0] == '1':
                    tempM[vidx] = tf.add(tempM[vidx],self.masks[m])
                elif m[0] == '2':
                    tempM[vidx] = tf.mul(tempM[vidx],self.masks[m])


            elif any([ (m[1:] in var and 'Matrix' in var) for var in vnames]):

                #Name of mask might be part of a variable scope

                for var in vnames:
                    if  (m[1:] in var and 'Matrix' in var):
                        vidx = vnames.index(var) #Index of variable
                if   m[0] == '0':
                    tempM[vidx] = self.masks[m]
                elif m[0] == '1':
                    tempM[vidx] = tf.add(tempM[vidx],self.masks[m])
                elif m[0] == '2':
                    tempM[vidx] = tf.mul(tempM[vidx],self.masks[m])


            elif m[-2:] == '_B' and any([ (m[1:] in var and 'Bias' in var) for var in vnames]):

                #Name of mask might be part of a variable scope

                for var in vnames:
                    if  m[1:] in var and 'Bias' in var and m[-2:] == '_B':
                        vidx = vnames.index(var) #Index of variable
                if   m[0] == '0':
                    tempM[vidx] = self.masks[m]
                elif m[0] == '1':
                    tempM[vidx] = tf.add(tempM[vidx],self.masks[m])
                elif m[0] == '2':
                    tempM[vidx] = tf.mul(tempM[vidx],self.masks[m])



            elif m[1:-1]+'W' in vnames:

                #Last char might be different to prevent confusion
                vidx = vnames.index(m[1:-1]+'W') #Index of variable

                if   m[0] == '0':
                    tempM[vidx] = self.masks[m]
                elif m[0] == '1':
                    tempM[vidx] = tf.add(tempM[vidx],self.masks[m])
                elif m[0] == '2':
                    tempM[vidx] = tf.mul(tempM[vidx],self.masks[m])

            vidxAll.append(vidx) 
            
        vidxAll = np.unique(vidxAll) #One value per var

        return [Vars[i].assign(tempM[i]) for i in vidxAll]

    def _VNoise(self, variables, learningR = .001, std = .001):
        ''' Adding stochasticity in the tensorflow graph variables at every update
           for a stotastic gradient descent.

            variables: list of all variables
            std      : standart deviation for normal distrubution'''

        V = []
        #Adding random noise
        for v in variables:
            V_add = v.assign_add( learningR*
                                  tf.random_normal(
                                  v.get_shape().as_list(), std) )
            V.append(V_add)

        return V




    def launchGraph(self, inputData, savepath = '_.ckpt'):
        # Launch the graph
        #   Arguments ... 
        #       inputData ~ Has to be a list of 4 elements : Training_X, training_Y
        #                                                    Testing_X , testing_Y.

        t = time.time() # Current time
        backupPath = '/tmp/backup.ckpt' # Checkpoint backup path

        #Unpacking data
        Xtr = inputData['Xtr']
        Ytr = inputData['Ytr'] 
        Xte = inputData['Xte']
        Yte = inputData['Yte']

        #Setting configs for minimum threads (small model)
        # config = tf.ConfigProto(device_count={"CPU": 56},
        #                  inter_op_parallelism_threads=1,
        #                  intra_op_parallelism_threads=1)

        # , config = config

        with tf.Session(graph=self.graph) as sess:

            #Initializing the variables
            sess.run(tf.initialize_all_variables())

            stepTr  = 0 #Training steps
            stepBat = 0 #Testing steps
            Loss    = 0 #Hold the last X points for mean Loss

            if self.sampRate > 0:
                samp   = 0                      #Sample
                nbSamp = self.nbIters//self.sampRate #Number of sample

            # Keep training until reach max iterations
            while stepTr < self.nbIters:

                if stepTr % self.dispStep == 0:
                    Xtr_batch, Ytr_batch = self.batchCreation(Xtr, Ytr, 
                                                      nbIters   = self.dispStep )

                stepBat = 0 #for batches based on display step

                feed_dict={  self._X    : Xtr_batch[stepBat,...], 
                             self._Y    : Ytr_batch[stepBat,...],
                             self._batch : stepTr  }

                #Running backprop
                sess.run(self.optimizer, feed_dict)
                
                #Applying masks
                #sess.run(self.masking)


                if stepTr % self.dispStep == 0:
                    testX,testY = self.batchCreation(Xte, Yte, 
                                                nbIters   = 1)

                    #Dictionnary for plotting training progress
                    feedDict_Tr = {  self._X    : Xtr_batch[stepBat,...], 
                                     self._Y    : Ytr_batch[stepBat,...],
                                     self._batch : stepTr  }

                    #Training fit (mean L2 loss)
                    trFit   = sess.run(self._ngml, feed_dict = feedDict_Tr)

                    #Weight sparsity
                    trSpars = sess.run(self._sparsC, feed_dict = feedDict_Tr)

                    #Testing fit with new data
                    teFit   = sess.run(self.precision, feed_dict={ self._X    : testX[0,...], 
                                                                   self._Y    : testY[0,...]  }) 

                    #Printing progress 

                    print( "Iter: " + str(stepTr) + "/" + str(self.nbIters)  +  
                           "   ~  L2 Loss: "      + "{:.6f}".format(trFit)   +
                           "   ~  L1 W loss: "    + "{:.6f}".format(trSpars) + 
                           "   |  Test fit: "     + "{:.6f}".format(teFit)    )

                if stepTr >= self.nbIters-50 :
                   Loss = Loss + teFit/50

                stepTr += 1
                stepBat += 1

                #Tracking variables in v2track
                if self.sampRate > 0 and not stepTr%self.sampRate:
                    self._trackVar( nbSamp, samp, self.v2track )
                    samp +=1

            # - - - - - - - -   - - - - -- - - - - - - - - - - -- - - 

            #Saving variables
            self.saver.save(sess, savepath)
            self.saver.save(sess, backupPath)
                        #Saving variables final state
            self.evalVars = {v.name: v.eval() for v in tf.trainable_variables()} 

            print('\nTotal time:  ' + str(datetime.timedelta(seconds = time.time()-t)))
            
        return Loss


    def _trackVar(self, nbSamp, samp, v2track):
        ''' 
        Will save the variables overtime at a given sampling rate.

        _______________________________________________________________

                                     ARGUMENTS
        _______________________________________________________________

        nbSamp  : total number of samples
        samp    : Current sample
        v2track : Contains the list of variables to track
        _______________________________________________________________

        '''

        #Values of variables of cu1rent sample
        vTrackVal = {v: self.vnames[v+':0'].eval() for v in v2track}

        #Initializing dictionnary
        if not hasattr(self, 'vTracked'):

            self.vTracked = {}
            for v in v2track:
                vdim = np.hstack([nbSamp,self.vnames[v+':0'].get_shape().as_list()])
                self.vTracked[v] = np.zeros(vdim)
        
        #Storing variables values at current sample
        for v in v2track:
            self.vTracked[v][samp,:] = vTrackVal[v]

        #Storing only direct connectivity
        if samp == nbSamp-1:
            tt = [('RNNCell' in var and 'Matrix' in var) for var in v2track]
            if any(tt):
                vidx = tt.index(True) #Variable index

                #If using __NGCmodel__ model
                if self.model == '__NGCmodel__':
                    dirC = self.vTracked[ v2track[vidx] ][: , self.nInput: -self.nhidGlob,
                                                            :-self.nhidGlob ]
                    self.vTracked['direct_W'] = dirC

        
    def showVars(self):
        #Will plot all variables

        idx   = 0 # Variable index
        spIdx = 0 # Subplot index

        for v in self.vnames:
            
            if spIdx % 4 == 0:
                plt.figure(figsize = (20,5))
                spIdx = 1
            
            #Dimension of variable
            dim = np.shape(self.evalVars[v])

            if len(dim)>1 and dim[0]>1 and dim[1]>1:

                #Plotting 2 dimensional variables
                plt.subplot(1,3,spIdx)
                
                #Plotting network H->H connectivity matrix only
                if 'BasicRNNCell' in v and 'Matrix' in v:
                      plt.imshow(self.evalVars[v][self.nhidNetw:,:], aspect = 'auto', interpolation = 'None')
                else:
                      plt.imshow(self.evalVars[v], aspect = 'auto', interpolation = 'None') 
                        
                plt.title(v[:-2], fontsize = 15)
                plt.colorbar()

            else:

                #Plotting one dimensional variables
                plt.subplot(1,3,spIdx)
                plt.plot(self.evalVars[v], 'x')
                plt.title(v[:-2], fontsize = 15)
                
            spIdx +=1
            idx   +=1


    def vidTrack(self, v2track = None, norm = True):
        ''' 
            Will diplay images of tracked variables in v2track
            over sample, giving a video. 


        _______________________________________________________________

                                    ARGUMENTS
        _______________________________________________________________

        nbSamp  : total number of samples
        samp    : Current sample
        v2track : Contains the list of variables to track
        _______________________________________________________________


        '''

        if not v2track:
            v2track = self.v2track

        #Function variables 
        nSample = self.vTracked[v2track[0]].shape[0] #Number of sample
        nVars   = len(v2track)                  #Number of variables to display
        nRow    = int(np.ceil(nVars/3))         #Number of subplot row

        # Number of collumns
        if   nVars  >= 3: nCol = 3 
        elif nVars  <  3: nCol = nVars 

    
        #Using only v2track variables
        vTracked = {v: self.vTracked[v].copy()  for v in v2track}  #Values
        dims     = {v: vTracked[v][0,...].shape for v in vTracked} #Dimensions

        #Normalizing each sample to see how things evolve
        if norm:
            for v in vTracked:
                for s in range(nSample):
                    vTracked[v][s,...] = vTracked[v][s,...]/abs(vTracked[v][s,...]).max()
                
        #Display 
        fig = plt.figure(figsize = (30,15))
        plt.hold(False)
        for s in range(nSample):

            spIdx = 1 # Subplot index
            for v in vTracked:
                ax = plt.subplot(nRow,nCol,spIdx)

                if len(dims[v])>1 and dims[v][0]>1 and dims[v][1]>1:
                    #Plotting 2 dimensional variables
                    plt.imshow(vTracked[v][s,...],
                               aspect = 'auto', clim=(-1,1),
                               interpolation = 'None') 
                    
                else:
                    #Plotting one dimensional variables
                    plt.plot(vTracked[v][s,:], 'x', ms = 20, mew = 3)
                    ax.set_ylim(-1,1)
                
                plt.title(v, fontsize = 15)
                spIdx += 1

                plt.draw()

            display.clear_output(wait=True)
            display.display(plt.gcf())

        
        display.clear_output()


def plotfit(paramFile, argDict= None, idx = range(1000), ckpt='/tmp/backup.ckpt'):
    ''' 
    1
    Will plot the real values and the fit of the model on top of it
    in order to evaluate the fit of the model.

    _________________________________________________________________________

                                   ARGUMENTS
    _________________________________________________________________________
        
         argDict   : Overwriting certain paramers. Has to be of type dict
         paramFile : Parameter file to run (in activeConnMain)
         nbPoints  : number of time points to pick from the fit dataset 
         ckpt      : ckpt can either be a full ckpt path, or only ckpt name.
                       ~> If only the name is provided, plotfit will look into 
                             activeConn/checkpoints/
                       ~> If agument not provided, plotfit will use the backup 
                             ckpt in /tmp.

    _________________________________________________________________________

    '''

    #Recovering main file function
    mainFile = getattr(ACM, paramFile)
    nbPoints = len(idx)

    argDict['batchSize'] = nbPoints #Adjusting batchsize for nb of points

    #Building graph
    graphFit, dataDict = mainFile(argDict, run = False)

    X = dataDict['FiX'][:,idx,:] # Input
    Y = dataDict['FiY'][idx,:]   # Label

    #Loading ckpt if checkpoint name only is provided
    if ckpt[0] != '/':
        ckpt = graphFit._mPath + 'checkpoints/' + ckpt

    #Passing data in model
    with tf.Session(graph=graphFit.graph) as sess:
        graphFit.saver.restore(sess, ckpt)
        with tf.variable_scope("") as scope:
            scope.reuse_variables()

            nEx  = X.shape[0]

            #X is transposed since it does not go through batch creation function
            _Z2 = sess.run(graphFit._Z2, feed_dict = { graphFit._X : X.transpose((1,0,2)) })

    #Plotting test set
    plt.figure(figsize=(20, 5))
    plt.imshow(Y.T, aspect='auto') ; plt.title('Real Data')
    plt.xlabel('Time (frames)') ; plt.ylabel('Neurons') ; plt.colorbar()

    #Plotting prediction on test set
    plt.figure(figsize=(20, 5))
    plt.imshow(_Z2.T, aspect='auto') ; plt.title('Model Predictions')
    plt.ylabel('Neurons') ; plt.xlabel('Time (frames)') ; plt.colorbar()

    #Printing the difference between data and prediction
    plt.figure(figsize=(20, 5))
    plt.imshow((Y-_Z2).T, aspect='auto') ; plt.title('Real - Model')
    plt.ylabel('Neurons') ; plt.xlabel('Time (frames)') ; plt.colorbar()

    plt.show()

    return graphFit, dataDict


#    def sensitivityA(self, data, ckpt='/tmp/backup.ckpt'):



