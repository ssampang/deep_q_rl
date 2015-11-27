#Replaces ale. Placeholder
from GoBoard import Board, BoardError, Location
from DCNN.DCNNGo import DCNNGo

import numpy as np

width = 19
boardErrors = 0
class GoReferee():
    def __init__(self, width):
        self.board = Board(width)
        self.dcnn = DCNNGo(Location('white'))

    def game_over(self):
        return False

    def reset_game(self):
        self.board = Board(width)



    def act(self, action):
        global boardErrors
        #must return an int reward
        posX = action / width
        posY = action % width
        try:
            self.board.move(posX, posY)
        except BoardError:
            boardErrors += 1
            print 'BoardErrors: '+str(boardErrors)
            #returning negative reward for illegal moves
            return -1

        #make DCNN move
        self.dcnn.placeStone(self.board, posX, posY)

        #First player in GoBoard is black, return black's score as reward
        return self.board.score['black']

    def getScreenGrayscale(self, npImageBuffer):
        #returning the array state
        for i in range(0,19):
            for j in range(0,19):
                if(self.board._array[i][j] == Location('black')):
                    npImageBuffer[i][j] = 1
                elif(self.board._array[i][j] == Location('white')):
                    npImageBuffer[i][j] = 0
                else:
                    npImageBuffer[i][j] = -1
        return npImageBuffer

    def getMinimalActionSet(self):
        #the universal set of moves the board can take
        #this is currently set to 361 as there are 361 board positions the stone can be placed on
        return range(0,361)
    
    def getScreenDims(self):
        return width,width 
