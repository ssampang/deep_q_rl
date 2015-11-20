
# coding: utf-8

# In[1]:

import json
import lasagne
from lasagne import nonlinearities
import numpy as np
import theano


# In[2]:

dataT = open("allenNetwork.json")
s = dataT.read()
obj = json.loads(s)


# In[3]:

listOfLayers = obj['layers']


# In[4]:

# build input layer
inputLayer = listOfLayers[0]
input_var = theano.tensor.tensor4('inputs')
l_in = lasagne.layers.InputLayer(shape=(None, 8, 25, 25), input_var=input_var)


# In[6]:

layerWeightMatrices = []
convolutionLayers = [l_in]

for layer in listOfLayers:
    if(layer['layer_type'] == 'conv'):
        filterWeights = layer['filters']
        
        noOfFilters = len(filterWeights) 
        sx = filterWeights[0]['sx']
        sy = filterWeights[0]['sy']
        depth = filterWeights[0]['depth']
        print depth, sx, sy, layer['stride'], layer['pad']
        weightArray = np.zeros((noOfFilters,depth,sx,sy))
        
        for index in range(0, len(filterWeights)):
            weights = np.array(filterWeights[index]['w'])
            weights = weights.reshape((depth,sx,sy))
            weightArray[index] = weights
        
        biases = np.array(layer['biases']['w'])
        
        convLayer = lasagne.layers.Conv2DLayer(incoming = convolutionLayers[-1], num_filters = noOfFilters, filter_size = (sx,sy), stride = layer['stride'], pad = layer['pad'], W = weightArray, b = biases, nonlinearity = lasagne.nonlinearities.rectify)
        convolutionLayers.append(convLayer)
        layerWeightMatrices.append(weightArray)


# In[7]:

for layer in listOfLayers:
    if(layer['layer_type'] == 'fc'):
        num_units  = len(layer['filters'])
        num_inputs = layer['num_inputs']
        weights = np.zeros((num_units, num_inputs))
        for i in range(0, num_units):
            weights[i] = np.array(layer['filters'][i]['w'])
        weights = np.transpose(weights)
        biases = np.array(layer['biases']['w'])
        fcLayer = lasagne.layers.DenseLayer(incoming = convolutionLayers[-1], num_units = num_units, W = weights, b = biases, nonlinearity=lasagne.nonlinearities.softmax)


# In[8]:

prediction_ = lasagne.layers.get_output(fcLayer, deterministic=True)
predict_function = theano.function([input_var], prediction_)


# In[9]:

#Test with an input
inputBoard = np.zeros((8,25,25))
# Set all squares in the edge channel to 1
for x in range(0,25):
    for y in range(0,25):
        if (x < 3 or x >= 22 or y < 3 or y >= 22):
            inputBoard[7][x][y] = 1


# In[10]:

inputBoard = inputBoard.reshape(1,8,25,25)
Y_pred = predict_function(inputBoard)
print Y_pred


# In[13]:

print np.argmax(Y_pred[0])


# In[16]:

print (281/19), (281 % 19)


# In[23]:

boardProbs = Y_pred[0].reshape((19,19))
print np.argmax(boardProbs)
print np.amax(boardProbs)


# In[25]:

print boardProbs[14][15]


# In[ ]:



