# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:22:27 2015

@author: adityanagarajan
"""


import subprocess 
import re
import numpy as np
import theano

class gnugo():
    def __init__(self,board_size = 19):
        self.proc = None
        self.color = 'black'
        self.gnugo_ctr = 1
        self.board_size = board_size
        self.move_white = None
        self.move_black = None
        self.score_white = 0
        self.score_black = 0
        self.board_state = np.zeros((3,self.board_size,self.board_size),dtype=theano.config.floatX)  
    
    def start_gnugo(self):
        self.proc = subprocess.Popen(['gnugo', '--mode', 'gtp'], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        self.proc.stdin.write(str(self.gnugo_ctr) + ' boardsize ' + str(self.board_size) + '\n')
        self.gnugo_ctr += 1
        start_game = self.get_gnugo_out()
        print start_game
    
    def close_gnugo(self):
        self.proc.stdin.write(str(self.gnugo_ctr) + 'quit' + '\n')
        self.gnugo_ctr = 1
    
    def set_learning_agent_color(self,color):
        self.color = color
    
    def get_score(self):
        self.proc.stdin.write(str(self.gnugo_ctr) + ' ' + 'captures white'+ '\n')
        self.gnugo_ctr +=1
        self.score_white = self.get_gnugo_out()
        self.proc.stdin.stdout.write(str(self.gnugo_ctr) + ' ' + 'captures black'+ '\n')
        self.gnugo_ctr +=1
        self.score_black = self.get_gnugo_out()
        
    def get_move_white(self):
        self.proc.stdin.write(str(self.gnugo_ctr) + ' ' + 'genmove white'+ '\n')
        self.gnugo_ctr +=1      
        self.white_move = self.get_gnugo_out()
        self.white_move = self.white_move.strip().split()[1]
        alpha,numa = (re.findall('[a-z|A-Z]',self.white_move)[0],re.findall('\d+',self.white_move)[0])
        
        self.white_move = self.map_from_board(alpha,numa)
        
        self.board_state[2,self.white_move[0],self.white_move[1]] = 1. 
        return self.white_move
    
    def get_move_black(self,b_x,b_y):
        self.move_black = (b_x,b_y)
        self.board_state[0,b_x,b_y] = 1.
        
        place_stone = self.map_to_board(b_x,b_y)
        self.proc.stdin.write(str(self.gnugo_ctr) + ' ' + 'play black ' + place_stone + '\n')
        self.gnugo_ctr +=1
        return self.get_gnugo_out()
    
    def map_to_board(self,x,y):
        alpha_numa_dict = {'A' : 0,
                      'B' : 1,
                      'C' : 2,
                      'D' : 3,
                      'E' : 4,
                      'F' : 5,
                      'G' : 6,
                      'H' : 7,
                      'I' : 8,
                      'J' : 9,
                      'K' : 10,
                      'L' : 11,
                      'M' : 12,
                      'N' : 13,
                      'O' : 14,
                      'P' : 15,
                      'Q' : 16,
                      'R' : 17,
                      'S' : 18}
        
        numa_alpha_dict = {v: k for k, v in alpha_numa_dict.items()}
        
        return numa_alpha_dict[x] + str(19 - y + 1)
        
    def map_from_board(self,alpha,numa):
        alpha_numa_dict = {'A' : 0,
                      'B' : 1,
                      'C' : 2,
                      'D' : 3,
                      'E' : 4,
                      'F' : 5,
                      'G' : 6,
                      'H' : 7,
                      'I' : 8,
                      'J' : 9,
                      'K' : 10,
                      'L' : 11,
                      'M' : 12,
                      'N' : 13,
                      'O' : 14,
                      'P' : 15,
                      'Q' : 16,
                      'R' : 17,
                      'S' : 18}
        x,y = (alpha_numa_dict[alpha],self.board_size - int(numa) -1)
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


obj = gnugo()

obj.start_gnugo()

a = obj.get_move_black(1,2)

a = obj.get_move_white()

a = obj.get_move_black(1,3)

a = obj.get_move_white()

a = obj.show_board()

obj.close_gnugo()

print obj.board_state


