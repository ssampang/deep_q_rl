# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:22:27 2015

@author: adityanagarajan
"""

from GoBoard import Board, BoardError, Location, View

import subprocess 
import re
import numpy as np
import theano
import sys

class gnugo():
    def __init__(self,board_size = 19,verbose = False):
        self.proc = None
        self.color = 'black'
        self.gnugo_ctr = 1
        self.board_size = board_size
        self.board_depth = 3
        self.board = Board(self.board_size)
        self.move_white = None
        self.move_black = None
        self.score_white = 0
        self.score_black = 0
        self.verbose = verbose
        self.board_state = np.zeros((self.board_depth,self.board_size,self.board_size),dtype=np.uint8)
        self.boardErrors = 0
        self.move_counter = 0
        self.white_pass = False
        
    
    def game_over(self):
        if self.move_counter % 200 == 0 or self.white_pass:
            return True
        else:
            return False
        
    def reset_game(self):
        self.move_counter = 0
        self.boardErrors = 0
        self.board = Board(self.board_size)
        self.white_pass = False
        self.close_gnugo()
    
    def getMinimalActionSet(self):
        #the universal set of moves the board can take
        #this is currently set to 361 as there are 361 board positions the stone can be placed on
        return range(0,self.board_size * self.board_size)
    
    def start_gnugo(self):
        self.board_state[0,:,:] = 0.
        self.board_state[2,:,:] = 0.
        self.board_state[1,:,:] = 0. 
        self.proc = subprocess.Popen(['gnugo', '--mode', 'gtp'], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        self.proc.stdin.write(str(self.gnugo_ctr) + ' boardsize ' + str(self.board_size) + '\n')
        self.gnugo_ctr += 1
        start_game = self.get_gnugo_out()
        print '-------Starting a new game--------'
    
    def close_gnugo(self):
        self.proc.stdin.write(str(self.gnugo_ctr) + 'quit' + '\n')
        self.gnugo_ctr = 1
    
    def set_learning_agent_color(self,color):
        self.color = color
    
    def get_score(self):
        self.proc.stdin.write(str(self.gnugo_ctr) + ' ' + 'captures white'+ '\n')
        self.gnugo_ctr +=1
#        score_white = self.score_white
        score_white = int(re.findall('\d+',self.get_gnugo_out())[1])
        self.proc.stdin.write(str(self.gnugo_ctr) + ' ' + 'captures black'+ '\n')
        self.gnugo_ctr +=1
        score_black = int(re.findall('\d+',self.get_gnugo_out())[1])
        
        reward = (score_black - self.score_black)
        if score_white - self.score_white > 0:
            print 'White captured move check reward -1'
            reward = -(score_white - self.score_white)
        self.score_black = score_black
        self.score_white = score_white
        return reward
    
#    def get_reward()
        
        
    def get_move_white(self):
        self.proc.stdin.write(str(self.gnugo_ctr) + ' ' + 'genmove white'+ '\n')
        self.gnugo_ctr +=1      
        self.move_white = self.get_gnugo_out()
        if 'PASS' not in self.move_white:
            self.move_white = self.move_white.strip().split()[1]
        
            alpha,numa = (re.findall('[a-z|A-Z]',self.move_white)[0],re.findall('\d+',self.move_white)[0])
        
            self.move_white = self.map_from_board(alpha,numa)
        
            self.board_state[2,self.move_white[0],self.move_white[1]] = 1. 
            self.board_state[1,self.move_white[0],self.move_white[1]] = 1. 
        else:
            # End game if white passes
            print 'White has passed the game!'
            self.move_white = None
            self.white_pass = True
        
        return self.move_white
    
    def get_move_black(self,b_x,b_y):
        self.move_black = (b_x,b_y)
        self.board_state[0,b_x,b_y] = 1.
        place_stone = self.map_to_board(b_x,b_y)
        self.proc.stdin.write(str(self.gnugo_ctr) + ' ' + 'play black ' + place_stone + '\n')
        self.gnugo_ctr +=1
        # Flush the output 
        temp = self.get_gnugo_out()
#        if 'illegal move' in temp:
#            print 'Illegal Move encountered exiting'
#            sys.exit()
        self.board_state[0,self.move_black[0],self.move_black[1]] = 1. 
        self.board_state[1,self.move_black[0],self.move_black[1]] = 1. 
        
        return self.move_black
    
    def map_to_board(self,x,y):
        alpha_numa_dict = {'A' : 0,
                      'B' : 1,
                      'C' : 2,
                      'D' : 3,
                      'E' : 4,
                      'F' : 5,
                      'G' : 6,
                      'H' : 7,
                      'J' : 8,
                      'K' : 9,
                      'L' : 10,
                      'M' : 11,
                      'N' : 12,
                      'O' : 13,
                      'P' : 14,
                      'Q' : 15,
                      'R' : 16,
                      'S' : 17,
                      'T' : 18}
        
        numa_alpha_dict = {v: k for k, v in alpha_numa_dict.items()}
        
        return numa_alpha_dict[x] + str(self.board_size - y)
        
    def map_from_board(self,alpha,numa):
        alpha_numa_dict = {'A' : 0,
                      'B' : 1,
                      'C' : 2,
                      'D' : 3,
                      'E' : 4,
                      'F' : 5,
                      'G' : 6,
                      'H' : 7,
                      'J' : 8,
                      'K' : 9,
                      'L' : 10,
                      'M' : 11,
                      'N' : 12,
                      'O' : 13,
                      'P' : 14,
                      'Q' : 15,
                      'R' : 16,
                      'S' : 17,
                      'T' : 18}
        x,y = (alpha_numa_dict[alpha],self.board_size - int(numa))
        return x,y

    def show_board(self):
        self.proc.stdin.write(str(self.gnugo_ctr) + ' ' + 'showboard' + '\n')
        return self.get_gnugo_out()
    
    def get_gnugo_out(self):
        s = ''
        r = self.proc.stdout.read(1)
        s += r
        while not r == '\n':
            r = self.proc.stdout.readline()
            s += r
        
        return s


    def place_filtered_moves(self,posX,posY):
        '''Once the moves have been verified legal, we pass them to this 
           function which places the stones on the board and gets gnugo's move'''
        move_b = self.get_move_black(posX,posY)
        
        move_w = self.get_move_white()
        
        reward = self.get_score()
        
        if self.verbose:
            print 'Move Number: %d'%self.move_counter
            if move_w != None:
                print 'Black plays: (%d,%d), White plays: (%d,%d)'%(move_b[0],move_b[1],move_w[0],move_w[1])
                print 'Black Score = %d, White Score = %d'%(int(self.score_black),int(self.score_white))
                print self.show_board()
        
        return reward
        
    def act(self, action):
        global boardErrors
        #must return an int reward
        posX = action % self.board_size
        posY = action / self.board_size
        try:
            self.board.move(posX, posY)
        except BoardError:
            self.boardErrors += 1
            print 'BoardErrors: '+str(self.boardErrors)
            #returning negative reward for illegal moves
            return -1
        
        
            
        reward = self.place_filtered_moves(posX,posY)
        if self.move_white == None:
            return 0
        
        self.board.move(self.move_white[0],self.move_white[1])
        
        self.move_counter += 1
        
        
#        view_ = View(self.board)
#        view_.redraw()
        
#        sys.stdout.write('{0}\n'.format(view_))
        
        #First player in GoBoard is black, return black's score as reward
        return reward

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
        



#obj = gnugo()
#
#obj.start_gnugo()
#obj.place_filtered_moves(0,0)
#obj.place_filtered_moves(1,3)
#obj.place_filtered_moves(1,4)
#obj.place_filtered_moves(5,4)
#obj.place_filtered_moves(9,4)
#obj.place_filtered_moves(18,18)
#
#print obj.show_board()
#
##obj.show_board()
#obj.close_gnugo()



#obj.start_gnugo()
#
#a = obj.get_move_black(1,2)
#
#a = obj.get_move_white()
#
#a = obj.get_move_black(1,3)
#
#a = obj.get_move_white()
#
#a = obj.show_board()
#
#obj.close_gnugo()
#
#print obj.board_state


