#Replaces ale. Placeholder
from GoBoard import Board, BoardError, Location, View
from DCNN.DCNNGo import DCNNGo

import sys
import numpy as np

width = 19
boardErrors = 0
class GoReferee():
    
    def __init__(self, width):
        self.board = Board(width)
        self.dcnn = DCNNGo(Location('white'))
        self.move_counter = 0
        self.view_ = View(self.board)

    def game_over(self):
        if self.move_counter > 200:
            return True
        else:
            return False

    def reset_game(self):
        self.move_counter = 0
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
        self.move_counter += 1
        
        self.view_.redraw()
        
        sys.stdout.write('{0}\n'.format(self.view_))
        sys.stdout.write('Black: {black} <===> White: {white}\n'.format(**self.board.score))
        sys.stdout.write('{0}\'s move... '.format(self.board.turn))
        
        #First player in GoBoard is black, return black's score as reward
        return self.board.score['black']

    def getScreenGrayscale(self, npImageBuffer):
        #returning the array state
        for i in range(0,19):
            for j in range(0,19):
                if(self.board._array[i][j] == Location('black')):
                    npImageBuffer[i][j] = 0
                elif(self.board._array[i][j] == Location('white')):
                    npImageBuffer[i][j] = 255
                else:
                    npImageBuffer[i][j] = 127
        return npImageBuffer

    def getMinimalActionSet(self):
        #the universal set of moves the board can take
        #this is currently set to 361 as there are 361 board positions the stone can be placed on
        return range(0,361)
    
    def getScreenDims(self):
        return width,width 
