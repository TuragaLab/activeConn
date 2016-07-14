import time
import datetime

import numpy             as np
import matplotlib.pyplot as plt

from activeConn.tools import *

import tensorflow as tf
from tensorflow.python.ops             import rnn, rnn_cell
from tensorflow.python.ops             import variable_scope        as vs
from tensorflow.python.ops.constant_op import constant  as const



class actConnGraph(object):
    # Default graph for Active Connectonic uncovering
    ''' 
    MODEL DESCRIPTION:
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    The model is constituted of three main part :

    1. The global cell 
    2. The network cell
    3. integrating previous cell with calcium filter of input

    The first cell is a low rank RNN with the goal of capturing 
    global dynamics (fast and slow). 

    The second cell tries to capture the real dynamic between 
    the units (population or neurons) observed in the data.

    The third component integrates previous cell and is also 
    influenced by the previous inputstep with calcium decay.

    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    ARGUMENTS:
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    
    nhid_glod  : Number of Global cell units
    nInput    : Number of input units
    nhidNetw  : Number of Network cell units
    nOut      : Number of output units
    seqLen    : Length of sequences
    weights    : Dictionnary containing the weights of each cells
    masks      : Mask applied to the corresponding weights after every batch
    actfct     : Activation function used in RNN
    batchSize : Number of examples in each batch
    learnRate  : Learning rate coefficient
    model      : Which model to use ...
                    '__multirnn_model__' : RNN(glob+netw) + calcium dynamic

    ...
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


    FUNCTIONS:
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    __VNoise__  : Adding stochasticity in the tensorflow graph variables  
                  at every update for a stotastic gradient descent. 
    
    __masking__ : Adding masks in the tensorflow graph that will 
                  be applied to the weights at every iteration.
    
    launchGraph : Will lauch the training of the model.

    plotfit     : Will plot the test set and the prediction of the model.

    showVars    : Will plot the variables with imshow (matrices) and plot (vectors)


                ...

    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    MODELS
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    List of available models:
    

    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    MASKS
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    Masks are used to constrain the weight space that can be learned
         which are multiplied with the weights after every optimization.
    
     netw_IH_HH : Direct connection (identity) for I-> H and all-to-all (except id) for H->H   
     netw_HO    : Direct connection (identity) 

     Define weights
         Hidden -> Hidden (HH) structure is the second par of the W matrix, 
         concatenanted with Input -> Hidden (IH). 
         
     netw_IH_HH : NetworkCell input  -> hidden (IH) & hidden -> hidden (HH) 
     netw_H0 :    NetworkCell hidden -> output (HO)
     glob_H0 :    NetworkCell hidden -> output (HO)

     The first character of a mask has a meaning :
    
                       0 : Mask replace the respective weight
                       1 : Mask is multipled with the respective weight
                       2 : Mask is added to the respective weight
    
     Mask operations will be executed in the same order

                ....

    '''   
     # ------------------------------------------------- Variables ------------------------------------------------- #

    def __init__(self, feat_dict ):

        #Default model parameters
        defaults = {  'learnRate': 0.0001, 'nbIters':10000, 'batchSize': 50, 
                      'dispStep'  :200, 'model': '__multirnn_model__',
                      'actfct':tf.tanh,'seqLen':10, 'method':1, 't2Dist':1   }      

        #Verifying if all inputs are provided in dictionnary and adding to object
        for key, val in defaults.items():
             if not key in feat_dict:
                setattr(self, key, val)

        #Assigining attributes from feat_dict
        for key, val in feat_dict.items():
                setattr(self, key, val)


        #Total number of hidden units
        self.nhid = self.nhidNetw + self.nhidGlob

        #Specifying model
        model = getattr(self, self.model)

        #Building the graph 
        graph = tf.Graph()
        with graph.as_default():    

            #Variable placeholders 
            self._T1    = tf.placeholder("float", [None, self.seqLen, self.nInput]) # Input at time t
            self._T2    = tf.placeholder("float", [None, self.nInput])               # Input at time t+1
            self.initG  = tf.placeholder("float", [None, self.nhidGlob])             # State of global cell
            self.initNG = tf.placeholder("float", [None, self.nhid])                  # State of netw+global cell
            #self.learnRate = tf.placeholder("float", [])                             # Learning rate for adaptive LR

            #Shape data
            _Z1 = shapeData(self._T1, self.seqLen, self.nInput)

            # --------------------------------------------------- Model --------------------------------------------------- #
            
            #Prediction using models
            self._Z2 = model(_Z1, initNG= self.initNG, initG= self.initG)

            #List of all variables 
            self.variables = tf.trainable_variables() 		

            #Variables assigned to each of the variable name 
            self.vnames    = {v.name:v for v in self.variables} 

            # Define loss and optimizer
            
            #Cost for connectivity in the network cell (H->H)
            #self.sparsC  = tf.add_n([ tf.nn.l2_loss( self.vnames['netw_IH_HH/BasicRNNCell/Linear/Matrix:0'][self.nhidNetw:,:] ) ])
            self.sparsC  = tf.add_n([tf.reduce_sum(tf.abs(v)) for v in self.variables]) 
            
            #Sum of square distance
            self.ngml = tf.reduce_sum(tf.pow(self._Z2 - self._T2, 2))/10
            #Prior
            self.ngprior = 0

            #Total Cost
            self.cost = (self.ngml + self.ngprior + self.sparsC) / (2*self.batchSize)

            #To test the precision of the network
            self.precision = tf.reduce_mean(tf.pow(self._Z2 - self._T2, 2))
 
            # Backpropagation
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learnRate).minimize(self.cost) 

            # Adding gaussian noise to variables updates
            #self.V_add_noise = self.__VNoise__(self.variables) # List of var.assign_add(noise) for all variables

            # Applying masking for restained connectivity
            self.masking = self.__masking__(self.variables)

            self.saver = tf.train.Saver()

        self.graph = graph

    def __multirnn_model__(self,_Z1, initNG= None, initG= None):
        #Full multi-layer model

        #Defining masks
        self.masks = {'1ng_IH_HH': 
                          np.vstack([ 
                                     np.ones([self.nInput, self.nhid],  dtype='float32'),
                                     np.hstack([ np.ones( [self.nhidNetw]*2,       dtype='float32')   
                                          -np.identity(self.nhidNetw,              dtype='float32'),
                                           np.zeros([self.nhidNetw,self.nhidGlob], dtype='float32') ]),
                                     np.ones([self.nhidGlob,self.nhid], dtype='float32')
                                    ]),

                      '2ng_IH_HH': tf.random_normal([self.nInput + self.nhid, self.nhid],
                                                    0.01) * self.learnRate
                     } 

        #Defining weights
        self.weights = { 'ng_H0_W' : weightInit([self.nhid,self.nOut], 'ng_HO_W' ) }

        #Defining biases
        self.biases  = { 'ng_H0_B' : weightInit(self.nOut, 'ng_H0_B') }  

        #Defining other variables
        self.alpha   = tf.get_variable("alpha",[self.nInput,1])

        #Network + Global dynamic cell (concatenated)
        ngCell = rnn_cell.BasicRNNCell(self.nhid, activation= self.actfct)
        ngCell = rnn_cell.MultiRNNCell([ngCell])

        #Network state initialization
        ngO = initNG
        
        # Initializing
        Z2 = 0  #Prediction

        with tf.variable_scope("ng_IH_HH") as scope:
            for i in range(self.seqLen):

                #Reusing variables for RNN
                if i == 1:
                  scope.reuse_variables()

                #Prediction error for time t
                ZD = Z2 - _Z1[i]  

                #Network + global cell
                ngO, ngS = ngCell(ZD, ngO)

                #NG to output cells
                ng_Z2 = tf.matmul(ngO, self.weights['ng_H0_W'] + self.biases['ng_H0_B'])

                #Prediction with calcium dynamic
                Z2 = self.actfct(tf.matmul(_Z1[i], self.alpha) + ng_Z2)

        return Z2


    def __dir_model__(self,_Z1, initNG= None, initG= None):
        #Building the model following the structure defined under actConnGraph class

        #Defining the weights
        self.weights = { 
                         'netw_dir_W' : weightInit([self.nInput]*2,            'netw_dir_W' ),
                         'glob_HO_W'  : weightInit([self.nhidGlob, self.nOut], 'glob_HO_W'  )
                        } 

        #Defining the biases
        self.biases =  {  
                         'netw_dir_B' : weightInit(self.nOut, 'netw_HO_B'),
                         'glob_HO_B'  : weightInit(self.nOut, 'glob_HO_B') 
                        } 

        #Defining masks
        self.masks   = {
                         '1netw_dir_M' : np.ones([self.nhidNetw]*2,  dtype= 'float32') - 
                                         np.identity(self.nhidNetw,  dtype= 'float32')
                        } 

        #Global dynamic cell
        globCell = rnn_cell.BasicRNNCell(self.nhidGlob, activation = self.actfct) #No initialization required

        #Global dynamic cell output
        globOut, globState = rnn.rnn(globCell, _Z1, initial_state=initG, dtype=tf.float32, scope = 'glob_IH_HH')

        #Hidden to Output
        glob_Z2 = tf.matmul(globOut[-1], self.weights['glob_HO_W']) + self.biases['glob_HO_B']

        #Direct netw connectivity
        netw_Z2 = self.actfct(tf.matmul( _Z1[-1], self.weights['netw_dir_W']) + self.biases['netw_dir_B'])

        #Applying calcium filter
        _Z2 = tf.matmul(_Z1[-1],self.alpha) + netw_Z2 + glob_Z2

        return _Z2

        vars = ['direct','whatever']

    def __VNoise__(self, variables, learningR = .001, std = .001):
        ''' Adding stochasticity in the tensorflow graph variables at every update
           for a stotastic gradient descent.

            variables: list of all variables
            std      : standart deviation for normal distrubution'''

        V = []
        #Adding random noise
        for v in variables:
            V_add = v.assign_add(learningR * tf.random_normal(v.get_shape().as_list(), std))
            V.append(V_add)

        return V


    def __masking__(self, Vars):
        ''' Will create the operations to update the weights

        The first character of a mask has a meaning :
        
                           0 : Mask replace the respective weight
                           1 : Mask is multipled with the respective weight
                           2 : Mask is added to the respective weight
        
         Mask operations will be executed in the same order '''

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
                    tempM[vidx] = tf.mul(tempM[vidx],self.masks[m])
                elif m[0] == '2':
                    tempM[vidx] = tf.add(tempM[vidx],self.masks[m])

            elif any([ (m[1:] and 'Matrix') in var for var in vnames]):
                #Name of mask might be part of a variable scope

                for var in vnames:
                    if (m[1:] and 'Matrix') in var:
                        vidx = vnames.index(var) #Index of variable

                if   m[0] == '0':
                    tempM[vidx] = self.masks[m]
                elif m[0] == '1':
                    tempM[vidx] = tf.mul(tempM[vidx],self.masks[m])
                elif m[0] == '2':
                    tempM[vidx] = tf.add(tempM[vidx],self.masks[m])
            
            elif m[1:-1]+'W' in vnames:
                #Last char might be different to prevent confusion
                vidx = vnames.index(m[1:-1]+'W') #Index of variable

                if   m[0] == '0':
                    tempM[vidx] = self.masks[m]
                elif m[0] == '1':
                    tempM[vidx] = tf.mul(tempM[vidx],self.masks[m])
                elif m[0] == '2':
                    tempM[vidx] = tf.add(tempM[vidx],self.masks[m])
                    
            vidxAll.append(vidx) 
            
        vidxAll = np.unique(vidxAll) #One value per var

        return [Vars[i].assign(tempM[i]) for i in vidxAll]


    def launchGraph(self, inputData, savepath = '_.ckpt', dispStep = 100, nbIters = 5000):
        # Launch the graph
        #   Arguments ... 
        #       inputData ~ Has to be a list of 4 elements : Training_t1, training_t2
        #                                                    Testing_t1 , testing_t2.

        t = time.time()

        backupPath = '/tmp/backup.ckpt'

        #Unpacking data
        T1  = inputData['T1']
        T2  = inputData['T2'] 
        Te1 = inputData['Te1']
        Te2 = inputData['Te2']

        #Setting configs for minimum threads (small model)
        config = tf.ConfigProto(device_count={"CPU": 56},
                         inter_op_parallelism_threads=1,
                         intra_op_parallelism_threads=1)

        with tf.Session(graph=self.graph, config = config) as sess:
            # Initialization

            #Initializing the variables
            sess.run(tf.initialize_all_variables())

            stepTr  = 0 #Training steps
            stepBat = 0 #Testing steps
            Ginit   = np.zeros((self.batchSize, self.nhidGlob)) # Global state initialization
            NGinit  = np.zeros((self.batchSize, self.nhid))

            # Keep training until reach max iterations
            while stepTr < nbIters:

                if stepTr % dispStep == 0:
                    T1_batch,T2_batch = batchCreation(T1, T2, 
                                                      nbIters   = dispStep, 
                                                      batchSize = self.batchSize, 
                                                      seqLen    = self.seqLen)

                stepBat = 0 #for batches based on display step

                feed_dict={self._T1: T1_batch[stepBat,...], 
                           self._T2: T2_batch[stepBat,...], 
                           self.initG: Ginit, self.initNG: NGinit}

                #Running backprop
                sess.run(self.optimizer, feed_dict)
                
                #Adding noise to weights
                #sess.run(self.V_add_noise)

                # Applying masks
                sess.run(self.masking)


                if stepTr % dispStep == 0:
                    testX,testY = batchCreation(Te1, Te2, nbIters=1, batchSize=self.batchSize, seqLen=self.seqLen)

                    #Dictionnary for plotting training progress
                    feedDict_Tr = { self._T1    : T1_batch[stepBat,...], 
                                    self._T2    : T2_batch[stepBat,...], 
                                    self.initG  : Ginit, 
                                    self.initNG : NGinit }

                    #Training fit (mean L2 loss)
                    trFit   = sess.run(self.ngml, feed_dict = feedDict_Tr)

                    #Weight sparsity
                    trSpars = sess.run(self.sparsC, feed_dict = feedDict_Tr)

                    #Testing fit with new data
                    teFit   = sess.run(self.precision, feed_dict={ self._T1    : testX[0,...], 
                                                                   self._T2    : testY[0,...], 
                                                                   self.initG  : Ginit, 
                                                                   self.initNG : NGinit  }) 

                    #Printing progress 
                    print("Iter: " + str(stepTr) + "/" + str(nbIters) +   "  ~  L2 Loss: "     + "{:.6f}".format(trFit) +
               "  ~  Sparsity loss: " + "{:.6f}".format(trSpars) + "        |  Testing fit: " + "{:.6f}".format(teFit))


                stepTr += 1
                stepBat += 1

            # - - - - - - - -   - - - - -- - - - - - - - - - - -- - - 

            #Saving variables
            self.saver.save(sess, savepath)
            self.saver.save(sess, backupPath)

            print('\nTotal time:  ' + str(datetime.timedelta(seconds = time.time()-t)))
            self.evalVars = [v.eval() for v in tf.trainable_variables()] 


    def plotfit(self, T1, T2, ckpt='/tmp/backup.ckpt'):
        ''' Will plot the real values and the fit of the model on top of it
             in order to evaluate the fit of the model.

             data: data to test. Must be of shape [nb_ex , seqLen , nb_units]
             ckpt: checkpoint file, including full path '''

        #Passing data in model
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, ckpt)
            with tf.variable_scope("") as scope:
                scope.reuse_variables()

                nEx  = T1.shape[0]

                Ginit  = np.zeros((nEx, self.nhidGlob),dtype='float32') # Global state initialization
                NGinit = np.zeros((nEx, self.nhid))

                _Z2 = sess.run(self._Z2, feed_dict = { self._T1    : T1,
                                                       self.initG  : Ginit,
                                                       self.initNG : NGinit } )

        #Plotting test set
        plt.figure(figsize=(20, 5))
        plt.imshow(T2.T, aspect='auto') ; plt.title('Real Data')
        plt.xlabel('Time (frames)') ; plt.ylabel('Neurons') ; plt.colorbar()

        #Plotting prediction on test set
        plt.figure(figsize=(20, 5))
        plt.imshow(_Z2.T, aspect='auto') ; plt.title('Model Predictions')
        plt.ylabel('Neurons') ; plt.xlabel('Time (frames)') ; plt.colorbar()

        #Printing the difference between data and prediction
        plt.figure(figsize=(20, 5))
        plt.imshow((T2-_Z2).T, aspect='auto') ; plt.title('Real - Model')
        plt.ylabel('Neurons') ; plt.xlabel('Time (frames)') ; plt.colorbar()

        plt.show()

  
    def showVars(self):
        #Will plot all variables

        idx   = 0 # Variable index
        spIdx = 0 # Subplot index

        for v in self.variables:
            
            if spIdx % 4 == 0:
                plt.figure(figsize = (20,5))
                spIdx = 1
            
            #Dimension of variable
            dim = np.shape(self.evalVars[idx])

            if len(dim)>1 and dim[0]>1 and dim[1]>1:

                #Plotting 2 dimensional variables
                plt.subplot(1,3,spIdx)
                
                #Plotting network H->H connectivity matrix only
                if v.name == 'netw_IH_HH/BasicRNNCell/Linear/Matrix:0':
                      plt.imshow(self.evalVars[idx][self.nhidNetw:,:], aspect = 'auto', interpolation = 'None')
                else:
                      plt.imshow(self.evalVars[idx], aspect = 'auto', interpolation = 'None') 
                        
                plt.title(v.name[:-2], fontsize = 15)
                plt.colorbar()

            else:

                #Plotting one dimensional variables
                plt.subplot(1,3,spIdx)
                plt.plot(self.evalVars[idx])
                plt.title(v.name[:-2], fontsize = 15)
                
            spIdx +=1
            idx   +=1


#    def sensitivityA(self, data, ckpt='/tmp/backup.ckpt'):




