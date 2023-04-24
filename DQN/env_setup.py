#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 21:23:16 2019

@author: ml
"""


import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

from sklearn import preprocessing
import pickle
import random


class EnvGrid:

    def __init__(self,num_horizontal_grids, num_vertical_grids):

        self.num_vertical_grids = num_vertical_grids
        self.num_horizontal_grids = num_horizontal_grids
        self.grid_neighbours = []
        self.current_state_list = []

    def get_closest_neighbour(self, grid_num):

        self.grid_neighbours = []

        top_border = [num for num in range(1, self.num_vertical_grids+1)]
        left_border = [num for num in range(1+self.num_vertical_grids, self.num_vertical_grids*(self.num_horizontal_grids-1),self.num_vertical_grids)]
        right_border = [num for num in range(2*self.num_vertical_grids, (self.num_vertical_grids*self.num_horizontal_grids),self.num_vertical_grids)]
        bottom_border = [num for num in range((self.num_vertical_grids*self.num_horizontal_grids)-self.num_vertical_grids+1,(self.num_vertical_grids*self.num_horizontal_grids)+1, 1)]

        if grid_num in top_border:
            if grid_num == 1:
                self.grid_neighbours.append([grid_num+1, grid_num+self.num_vertical_grids, grid_num+self.num_vertical_grids+1])

            elif grid_num == self.num_vertical_grids:
                self.grid_neighbours.append([grid_num - 1, grid_num+self.num_vertical_grids, grid_num+self.num_vertical_grids-1])
            else:
                self.grid_neighbours.append([grid_num - 1, grid_num + 1,grid_num+self.num_vertical_grids, grid_num+self.num_vertical_grids-1, grid_num+self.num_vertical_grids+1])

        elif grid_num in left_border:
            self.grid_neighbours.append([grid_num+1, grid_num-self.num_vertical_grids, grid_num-self.num_vertical_grids+1, grid_num+self.num_vertical_grids, grid_num+self.num_vertical_grids+1])

        elif grid_num in right_border:
            self.grid_neighbours.append([grid_num-1, grid_num-self.num_vertical_grids, grid_num-self.num_vertical_grids-1, grid_num+self.num_vertical_grids, grid_num+self.num_vertical_grids-1])
        
        elif grid_num in bottom_border:
            if grid_num == (self.num_vertical_grids*(self.num_horizontal_grids-1))+1:
                self.grid_neighbours.append([grid_num+1,grid_num-self.num_vertical_grids, grid_num-self.num_vertical_grids+1])
            elif grid_num == self.num_vertical_grids*self.num_horizontal_grids:
                self.grid_neighbours.append([grid_num-1,grid_num-self.num_vertical_grids, grid_num-self.num_vertical_grids-1])
            else:
                self.grid_neighbours.append([grid_num-1,grid_num+1, grid_num-self.num_vertical_grids, grid_num-self.num_vertical_grids+1,grid_num-self.num_vertical_grids-1])

        else:
            self.grid_neighbours.append([
                grid_num+1, 
                grid_num-1,
                grid_num+self.num_vertical_grids, 
                grid_num+self.num_vertical_grids+1,
                grid_num+self.num_vertical_grids-1,
                grid_num-self.num_vertical_grids, 
                grid_num-self.num_vertical_grids+1,
                grid_num-self.num_vertical_grids-1
                ])
                

        return self.grid_neighbours

    def plot_grids(self):

        fig, ax = plt.subplots(figsize=(self.num_horizontal_grids, self.num_vertical_grids))
        ax.set_xlim(0, self.num_vertical_grids)
        ax.set_ylim(self.num_horizontal_grids, 0)
        ax.grid(True)
        ax.set_xticks(np.arange(0, self.num_vertical_grids+1, 1.0))
        ax.set_yticks(np.arange(0, self.num_horizontal_grids+1, 1.0))


        # create tuples of positions
        positions = [(x + 0.05, y + 0.5) for y in range(self.num_horizontal_grids) for x in range(self.num_vertical_grids)]


        grid_num = 1
        for x,y in positions:
            ax.text(x,y, grid_num, color = "blue")
            grid_num+=1


        plt.show()

    def reset(self):
        """ 
        Set/reset environment
        valid cells are any cells in the grid except right most column and bottom row
        returns list of grid numbers and normalized form of this. 
        first element is the elment of interest and the rest are the neighbourign cells
        """
        total_cells = [num for num in range(1, (self.num_horizontal_grids*self.num_vertical_grids)+1,1)]

        right_border = [num for num in range(self.num_vertical_grids, (self.num_vertical_grids*self.num_horizontal_grids),self.num_vertical_grids)]

        bottom_border = [num for num in range((self.num_vertical_grids*self.num_horizontal_grids)-self.num_vertical_grids+1,(self.num_vertical_grids*self.num_horizontal_grids)+1, 1)]

        valid_cells = set(total_cells)-set(right_border)-set(bottom_border)

        valid_cells = np.array(list(valid_cells))


        sel_state = np.random.choice(valid_cells)



        state_list = []

        neighbour_cells = self.get_closest_neighbour(sel_state)


        state_list = neighbour_cells[0]
        state_list.append(sel_state)

        state_list.reverse()
        self.current_state_list = state_list

        state_list_array = np.asarray(state_list)
        normalized_state_list = state_list_array/(np.float(self.num_vertical_grids * self.num_horizontal_grids))

        if len(state_list) < 9:
            for _ in range(9-len(state_list)):
                normalized_state_list = np.append(normalized_state_list,[0])

        
        return np.asarray(normalized_state_list)


    def env_behaviour(self, state_list, action):

        if action == self.current_state_list[0] + (self.num_vertical_grids+1):
            reward = 0
            done = True
            next_state = np.asarray([-1,-1,-1,-1,-1,-1,-1,-1,-1])
        else:
            reward = -1
            done = False
            next_state = state_list

        return next_state, reward, done
    


class EnvWeb:

    def __init__(self):

        with open('../integration/data/element_dictionary.pkl', 'rb') as f:
            self.element_dict = pickle.load(f)
        

        self.input_grid_num = np.asarray([self.element_dict[num]['input']['grid_num'] for num in range(1,len(self.element_dict)+1)])
        self.label_grid_num = [self.element_dict[num]['label']['grid_num'] for num in range(1,len(self.element_dict)+1)]

        self.index_list = [num for num in range(len(self.label_grid_num))]
        self.rand_label_index = None
            
    def reset(self):
        self.rand_label_index = random.choice(self.index_list)
        label = np.asarray(self.input_grid_num[self.rand_label_index])
        state = np.append(label, self.input_grid_num)
        
        return state
    
    def reset_test(self):
        """ outputs all possible states. used for testing. donot use for training"""
        states = []
        for item in self.label_grid_num:
            state = np.append(item, self.input_grid_num)
            states.append(state)
        return states    

    def env_behaviour(self, state_list, action):

        
        expected_action = self.input_grid_num[self.rand_label_index]
        

        if expected_action == action:
            reward = 0
            done = True
            next_state = np.asarray([-1,-1,-1,-1,-1,-1,-1,-1])
        else:
            reward = -1
            done = False
            next_state = state_list

        return next_state, reward, done

