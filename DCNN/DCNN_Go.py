import json
import lasagne
from lasagne import nonlinearities
import numpy as np
import theano

class DCNN_Network:

	def __init__(self):
		self.prediction_function = buildNetwork(networkPath)
		self.dcnnInput = np.zeros(8, 25, 25)
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
	
	def placeStone(board, oppenentMoveX, oppenentMoveX):
		#Update the liberty channels of the previous move of DCNN 
		if(self.moveX != -1 and self.moveY != -1):
			opponentLiberties = board.get_liberties(self.moveX, self.moveY)
			if(opponentLiberties > 3):
				opponentLiberties = 3
			self.dcnnInput[opponentLiberties] = 1
		#Update the liberty channels of the move by opponent 
		opponentLiberties = board.get_liberties(oppenentMoveX, oppenentMoveY)
		if(opponentLiberties > 3):
			opponentLiberties = 3
		self.dcnnInput[2 + opponentLiberties] = 1
		
		# TODO: setting ko channel remaining
		
		inputToDCNN = self.dcnnInput.reshape(1,8,25,25)
		predProbs = self.prediction_function(inputToDCNN)
		pos = np.argmax(predProbs[0])
		self.moveX = pos / 19
		self.moveY = pos % 19
		return (self.moveX, self.moveY)

def buildNetwork(networkPath):
	dataT = open("allenNetwork.json")
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
			weightArray = np.zeros((noOfFilters,depth,sx,sy))
			
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