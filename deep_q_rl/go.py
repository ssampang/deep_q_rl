#Replaces ale. Placeholder
from .board import *

def __init__(self, width):
	self.board = Board(width)
	self.dcnn = DCNN_Network()

def game_over():
	return false

def reset_game() 
	self.board = Board(width)

boardErrors = 0
def act(action):
	global boardErrors
	#must return an int reward
	posX = action / (width*width)
	posY = action % (width*width)
	try:
		self.board.move(posX, posY)
		#make DCNN move
		self.dcnn.placeStone(self.board, posX, posY)
		#listen to Board exceptions
	except BoardError:
		boardErrors += 1

def getaScreenGrayscale:
	#returning the array state
	return self.board._array
	
def getMinimalActionSet():
	#the universal set of moves the board can take
	#this is currently set to 361 as there are 361 board positions the stone can be placed on
	return range(0,361)
