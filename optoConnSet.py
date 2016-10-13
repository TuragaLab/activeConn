# coding: utf-8

# Imports --------------------------------------------------------------------------

import numpy             as np
import matplotlib.pyplot as plt

from os.path         import expanduser 
from scipy.io        import loadmat
from optoConn.graphs import *
from optoConn.tools  import *

import tensorflow as tf


def optoConn(paramDict, data, run = True, graph= None):
    '''
    Bayesian RNN for Active Connectomics Learning
           
         Bayesian neural network that allows the 
         prediction of causal connectivity coupled with 
         pertubation of the system for stronger inferences. 
    _______________________________________________________________

                            ARGUMENTS
    _______________________________________________________________

    paramDict: Contains all the parameters passed into graph,
               which will be unpacked and added to class attributes
    data     : Data matrix of size (nb_input_units x time)
    run      : Will run the model or not
    _______________________________________________________________


    '''

    #Default model parameters
    pDict = {
               'dataset':'FR_RNN.mat', '_mPath': expanduser("~") + '/.optoConn/',
               'saveName': 'ckpt.ckpt', 'learnRate': 0.0001, 'nbIters':10000,
               'batchSize': 50, 'dispStep':200, 'model': '__NGCmodel__', 'detail':True,
               'actfct':tf.tanh,'prepMethod':1, 'YDist':1 , 'sampRate':0             
             }       

    #Updatating pDict with input dictionnary
    pDict.update(paramDict)

    # Path
    savepath = pDict['_mPath'] + 'checkpoints/' + pDict['saveName'] # Checkpoint save path


    #Formatting data
    if type(data) is dict:
        if data['dataset'].shape[0] != pDict['nInput']:
           data['dataset'] = data['dataset'][:pDict['nInput'],:]
           data['baseline'] = data['baseline'][:pDict['nInput'],:]

        if 'class' in pDict['model']:
            pDict['nInput']    = 1
            pDict['batchSize'] = pDict['batchSize']*2
            dataDict = dataPrepClassi( data,
                                       ctrl       = pDict['ctrl'],
                                       cells      = pDict['cells'],
                                       null       = pDict['null'] ,
                                       seqRange   = pDict['seqRange'], 
                                       prepMethod = pDict['prepMethod']  )
        else:
            dataDict = dataPrepGenerative( data, 
                                           seqRange   = pDict['seqRange'],
                                           prepMethod = pDict['prepMethod'],
                                           null       = pDict['null'] )

        tempD = {key: data[key] for key in data if key not in 'dataset'}
        dataDict.update(tempD)
    else:
        dataDict = dataPrepGenerative( data, 
                                       seqRange     = pDict['seqRange'],
                                       prepMethod   = pDict['prepMethod'] 
                                       )

    #InputSize warming
    if type(data) is not dict:
        nInput = dataDict['Xtr'].shape[2]
        if nInput != pDict['nInput']:
            raise ValueError('Number of input units in'       +
                            ' the data ({0})'.format(nInput)  +
                            ' and number of input units in'   +
                            ' the model ({0})'.format(pDict['nInput']) +
                            ' need to be equal.')

    #Build graph
    #print('Building Graph    ...')
    if not graph:
      graph = actConnGraph(pDict)

    #Launch Graph
    if run:
        #print('Launching Session ...')
        Acc = graph.launchGraph( dataDict, detail = pDict['detail'], savepath = savepath )
    else:
        Acc = None

    return graph, dataDict, Acc

