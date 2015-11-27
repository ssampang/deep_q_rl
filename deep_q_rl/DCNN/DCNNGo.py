import json
import lasagne
import numpy as np
import theano
import os
from GoBoard import BoardError

class DCNNGo():

    def __init__(self, boardColor):
        self.prediction_function = buildNetwork(os.path.join(os.path.dirname(__file__), "allenNetwork.json"))
        self.dcnnInput = np.zeros((8, 25, 25))
        for x in range(0,3):
            self.dcnnInput[7,x,:] = 1
            self.dcnnInput[7,:,x] = 1
        for x in range(22,25):
            self.dcnnInput[7,x,:] = 1
            self.dcnnInput[7,:,x] = 1
        for x in range(0,3):
            self.dcnnInput[7,x,:] = 1
            self.dcnnInput[7,:,x] = 1
        for x in range(22,25):
            self.dcnnInput[7,x,:] = 1
            self.dcnnInput[7,:,x] = 1
        
        #Storing the previous input to recompute liberties
        self.moveX = -1
        self.moveY = -1
        self.boardColor = boardColor

    def placeStone(self, board, oppenentMoveX, oppenentMoveY):
        
        self._buildLiberties(board, oppenentMoveX, oppenentMoveY)
        
        # Setting ko channel
        # Naive method. Going over all positions and checking
        for x in range(0,19):
            for y in range(0,19):
                if(board.checkKo(x,y) == True):
                    self.dcnnInput[6,x+3,y+3] = 1
        
        inputToDCNN = self.dcnnInput.reshape(1,8,25,25)
        predProbs = self.prediction_function(inputToDCNN)
        sortedPos = np.argsort(predProbs[0])
        for pos in sortedPos:
            self.moveX = pos / 19
            self.moveY = pos % 19
            try:
                board.move(self.moveX, self.moveY)
            except BoardError:
                continue
    
    def _buildLiberties(self, board, oppenentMoveX, oppenentMoveY):
        #Update the liberty channels at the previous move of DCNN 
        if(self.moveX != -1 and self.moveY != -1):
            dcnnLiberties = board.count_individual_liberties(self.moveX, self.moveY)[1]
            if(dcnnLiberties > 3):
                dcnnLiberties = 3
            self.dcnnInput[dcnnLiberties, self.moveX + 3, self.moveY + 3] = 1
        
        #Update the liberty channels in the neighbourhood of the previous move of DCNN 
        if(self.moveX > 0):
            dcnnLiberties = board.count_individual_liberties(self.moveX - 1, self.moveY)
            if(dcnnLiberties[1] > 3):
                dcnnLiberties[1] = 3
            if(dcnnLiberties[0] != self.boardColor):
                dcnnLiberties[1] += 2
            
            self.dcnnInput[dcnnLiberties[1], self.moveX - 1 + 3, self.moveY + 3] = 1
        if(self.moveX < 18):
            dcnnLiberties = board.count_individual_liberties(self.moveX + 1, self.moveY)
            if(dcnnLiberties[1] > 3):
                dcnnLiberties[1] = 3
            if(dcnnLiberties[0] != self.boardColor):
                dcnnLiberties[1] += 2
            
            self.dcnnInput[dcnnLiberties[1], self.moveX + 1 + 3, self.moveY + 3] = 1
        if(self.moveY > 0):
            dcnnLiberties = board.count_individual_liberties(self.moveX, self.moveY - 1)
            if(dcnnLiberties[1] > 3):
                dcnnLiberties[1] = 3
            if(dcnnLiberties[0] != self.boardColor):
                dcnnLiberties[1] += 2
            
            self.dcnnInput[dcnnLiberties[1], self.moveX + 3, self.moveY - 1 + 3] = 1
        if(self.moveY < 18):
            dcnnLiberties = board.count_individual_liberties(self.moveX, self.moveY + 1)
            if(dcnnLiberties[1] > 3):
                dcnnLiberties[1] = 3
            if(dcnnLiberties[0] != self.boardColor):
                dcnnLiberties[1] += 2
            
            self.dcnnInput[dcnnLiberties[1], self.moveX + 3, self.moveY + 1 + 3] = 1
        
        #Update the liberty channels in the neighbourhood of the previous move of opponent
        if(oppenentMoveX > 0):
            dcnnLiberties = board.count_individual_liberties(oppenentMoveX - 1, oppenentMoveY)
            if(dcnnLiberties[1] > 3):
                dcnnLiberties[1] = 3
            if(dcnnLiberties[0] != self.boardColor):
                dcnnLiberties[1] += 2
            
            self.dcnnInput[dcnnLiberties[1], oppenentMoveX - 1 + 3, oppenentMoveY + 3] = 1
        if(oppenentMoveX < 18):
            dcnnLiberties = board.count_individual_liberties(oppenentMoveX + 1, oppenentMoveY)
            if(dcnnLiberties[1] > 3):
                dcnnLiberties[1] = 3
            if(dcnnLiberties[0] != self.boardColor):
                dcnnLiberties[1] += 3
            
            self.dcnnInput[dcnnLiberties[1], oppenentMoveX + 1 + 3, oppenentMoveY + 3] = 1
        if(oppenentMoveY > 0):
            dcnnLiberties = board.count_individual_liberties(oppenentMoveX, oppenentMoveY - 1)
            if(dcnnLiberties[1] > 3):
                dcnnLiberties[1] = 3
            if(dcnnLiberties[0] != self.boardColor):
                dcnnLiberties[1] += 2
            
            self.dcnnInput[dcnnLiberties[1], oppenentMoveX + 3, oppenentMoveY - 1 + 3] = 1
        if(oppenentMoveY < 18):
            dcnnLiberties = board.count_individual_liberties(oppenentMoveX, oppenentMoveY + 1)
            if(dcnnLiberties[1] > 3):
                dcnnLiberties[1] = 2
            if(dcnnLiberties[0] != self.boardColor):
                dcnnLiberties[1] += 2
            
            self.dcnnInput[dcnnLiberties[1], oppenentMoveX + 3, oppenentMoveY + 1 + 3] = 1
        
        #Update the liberty channels of the move by opponent 
        opponentLiberties = board.count_liberties(oppenentMoveX, oppenentMoveY)[1]
        if(opponentLiberties > 3):
            opponentLiberties = 3
        self.dcnnInput[2 + opponentLiberties] = 1

def buildNetwork(networkPath):
    dataT = open(networkPath)
    s = dataT.read()
    obj = json.loads(s)
    
    listOfLayers = obj['layers']
    
    #initialize input layer
    inputLayer = listOfLayers[0]
    input_var = theano.tensor.tensor4('inputs')
    l_in = lasagne.layers.InputLayer(shape=(None, 8, 25, 25), input_var=input_var)
    
    #initialize convolution layers
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
            weightArray = np.zeros((noOfFilters,depth,sx,sy),dtype=theano.config.floatX)
            
            for index in range(0, len(filterWeights)):
                weights = np.array(filterWeights[index]['w'])
                weights = weights.reshape((depth,sx,sy))
                weightArray[index] = weights
            
            biases = np.array(layer['biases']['w'])
            
            convLayer = lasagne.layers.Conv2DLayer(incoming = convolutionLayers[-1], num_filters = noOfFilters, filter_size = (sx,sy), stride = layer['stride'], pad = layer['pad'], W = weightArray, b = biases, nonlinearity = lasagne.nonlinearities.rectify)
            convolutionLayers.append(convLayer)
            layerWeightMatrices.append(weightArray)
    
    #initialize fully connected layer
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
    
    prediction_ = lasagne.layers.get_output(fcLayer, deterministic=True)
    return theano.function([input_var], prediction_)
