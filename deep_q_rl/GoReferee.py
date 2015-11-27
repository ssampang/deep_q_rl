#Replaces ale. Placeholder
from GoBoard import Board
from GoBoard import BoardError
from DCNN.DCNNGo import DCNNGo

import numpy as np

width = 19
class GoReferee():
    def __init__(self, width):
        self.board = Board(width)
        self.dcnn = DCNNGo('')

    def game_over(self):
        return False

    def reset_game(self):
        self.board = Board(width)

    boardErrors = 0

    def act(self, action):
        global boardErrors
        #must return an int reward
        posX = action / (width*width)
        posY = action % (width*width)
        try:
            self.board.move(posX+1, posY+1)
        except BoardError:
            boardErrors += 1
            #returning negative reward for illegal moves
            return -1

        #make DCNN move
        self.dcnn.placeStone(self.board, posX, posY)

        #First player in GoBoard is black, return black's score as reward
        return self.board.score()['black']

    def getScreenGrayscale(self, npImageBuffer):
        #returning the array state

        return np.copyto(npImageBuffer, self.board._array)

    def getMinimalActionSet(self):
        #the universal set of moves the board can take
        #this is currently set to 361 as there are 361 board positions the stone can be placed on
        return range(0,361)
    
    def getScreenDims(self):
        return width,width 
