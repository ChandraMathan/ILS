
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

    def __init__(self, input_dict, state_size, grid):

        """
        params:

        input_dict: contains dictionary of input field location, label location and thier grid numbers.
        grid (tuple): horizontal and vertical grids 
        """

        with open(input_dict, 'rb') as f:
            self.element_dict = pickle.load(f)
        
        self.state_size = state_size

        self.grid = grid
        self.grid_vertical = self.grid[0]
        self.grid_horizontal = self.grid[1]

        self.input_grid_num = {}
        self.label_grid_num = {}

        self.webpages = []
        self.curr_webpage = None

        self.index_list = None 
        self.rand_label_index = None

        self.state_unnormalized = None
        self.state_normalized = None

        for item in self.element_dict:
            
            self.input_grid_num[item] = np.asarray([self.element_dict[item][str(num)]['input']['grid_num'] for num in range(1,len(self.element_dict[item])+1)])
            self.label_grid_num[item] = [self.element_dict[item][str(num)]['label']['grid_num'] for num in range(1,len(self.element_dict[item])+1)]
            
            self.webpages.append(item)
                
    def reset(self):
        
        self.curr_webpage = random.choice(self.webpages)
        self.index_list = [num for num in range(len(self.label_grid_num[self.curr_webpage]))]
        self.rand_label_index = random.choice(self.index_list)
        label = np.asarray(self.label_grid_num[self.curr_webpage][self.rand_label_index])
        state = np.append(label, self.input_grid_num[self.curr_webpage])

        if len(state)< self.state_size:
            for _ in range(0, self.state_size - len(state)):
                state = np.append(state, 0)

        self.state_unnormalized = state
        self.state_normalized = state/(self.grid_vertical*self.grid_horizontal) #note this normalization is based on min = 0, and (x-x_min)/(x_max - x_min)
        
        return self.state_normalized
    
    def reset_test(self):
        """ outputs all possible states and corresponding actions. used for testing. donot use for training"""
        
        all_states ={}
        for webpage in self.label_grid_num:
            states = []
            for item in self.label_grid_num[webpage]:
                state = np.append(item, self.input_grid_num[webpage])
                if len(state)< self.state_size:
                    for _ in range(0, self.state_size - len(state)):
                        state = np.append(state, 0)
                states.append(state)
            all_states[webpage] = states

        actions = {}
        for webpage in all_states:
            action_list = []
            for item in all_states[webpage]:
                label = item[0]
                if label in self.label_grid_num[webpage]:
                    label_index = self.label_grid_num[webpage].index(label)
                    expected_action = self.input_grid_num[webpage][label_index]
                else:
                    expected_action = -1
                action_list.append(expected_action)
            actions[webpage] = action_list
        
        return all_states, actions    

    def env_behaviour(self, action):

        #label = state_list[0]
        label = self.state_unnormalized[0]
        if label in self.label_grid_num[self.curr_webpage]:
            label_index = self.label_grid_num[self.curr_webpage].index(label)
            expected_action = self.input_grid_num[self.curr_webpage][label_index]
        else:
            expected_action = -1


        if expected_action == action:
            reward = 0.1
            done = True
            #next_state = np.asarray([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])
            next_state = np.full((self.state_size, ), -1.0)
            
               
        else:
            reward = -0.1
            done = False
            next_state = self.state_normalized
            

        return next_state, reward, done


class EnvWeb2D:

    """ 
    this environment is based on state shown as a single vector of x locations and y locations. 
    actions are specified as grids

    """

    def __init__(self, input_dict, state_size, grid, window_size):

        """
        params:

        input_dict: contains dictionary of input field location, label location and thier grid numbers.
        grid (tuple): horizontal and vertical grids 
        window_size (int): width and height of the images
        """

        with open(input_dict, 'rb') as f:
            self.element_dict = pickle.load(f)
        
        self.state_size = state_size
        self.window_size = window_size

        self.grid = grid
        self.grid_vertical = self.grid[0]
        self.grid_horizontal = self.grid[1]

        self.input_grid_num = {}
        self.label_grid_num = {}

        self.input_xy_location = {}
        self.input_x_location = {}
        self.input_y_location = {}

        self.label_xy_location = {}
        self.label_x_location = {}
        self.label_y_location = {}

        self.webpages = []
        self.curr_webpage = None

        self.index_list = None 
        self.rand_label_index = None

        self.state_unnormalized = None
        self.state_normalized = None

        #setup all states and actions in a dictionary

        for item in self.element_dict:
            
            self.input_grid_num[item] = np.asarray([self.element_dict[item][str(num)]['input']['grid_num'] for num in range(1,len(self.element_dict[item])+1)])
            self.label_grid_num[item] = np.asarray([self.element_dict[item][str(num)]['label']['grid_num'] for num in range(1,len(self.element_dict[item])+1)])

            self.input_x_location[item] = np.asarray([self.element_dict[item][str(num)]['input']['x_centre'] for num in range(1,len(self.element_dict[item])+1)])
            self.input_y_location[item] = np.asarray([self.element_dict[item][str(num)]['input']['y_centre'] for num in range(1,len(self.element_dict[item])+1)])

            self.label_x_location[item] = np.asarray([self.element_dict[item][str(num)]['label']['x_centre'] for num in range(1,len(self.element_dict[item])+1)])
            self.label_y_location[item] = np.asarray([self.element_dict[item][str(num)]['label']['y_centre'] for num in range(1,len(self.element_dict[item])+1)])

            self.webpages.append(item)

        self.state_action_dict = {}

        for item in self.label_x_location:
            item_index_state = []
            self.state_action_dict[item] = {}
            self.state_action_dict[item]['state'] = {}
            self.state_action_dict[item]['action'] = {}

            for index in range(0,len(self.label_x_location[item])):
                self.state_action_dict[item]['state'][index] = {}
                item_index_state = np.concatenate(([self.label_x_location[item][index]],self.input_x_location[item], [self.label_y_location[item][index]],self.input_y_location[item]),axis = None)
                self.state_action_dict[item]['state'][index] = item_index_state
                self.state_action_dict[item]['action'][index] = self.input_grid_num[item][index]
        
    def reset(self):
        
        self.curr_webpage = random.choice(self.webpages)
        self.index_list = [num for num in range(len(self.state_action_dict[self.curr_webpage]['state']))]
        self.rand_label_index = random.choice(self.index_list)
        #label = np.asarray(self.label_grid_num[self.curr_webpage][self.rand_label_index])
        #state = np.append(label, self.input_grid_num[self.curr_webpage])
        state = self.state_action_dict[self.curr_webpage]['state'][self.rand_label_index]

        if len(state)< self.state_size:
            for _ in range(0, self.state_size - len(state)):
                state = np.append(state, 0)

        self.state_unnormalized = state
        self.state_normalized = state/max(self.window_size) #note this normalization is based on min = 0, and (x-x_min)/(x_max - x_min)
          
        return self.state_normalized

    def env_behaviour(self, action):

        expected_action = self.state_action_dict[self.curr_webpage]['action'][self.rand_label_index]

        if expected_action == action:
            reward = 0.1
            done = True
            next_state = np.full((self.state_size, ), -1.0)
        
        else:
            reward = -0.1
            done = False
            next_state = self.state_normalized
            

        return next_state, reward, done

    def get_data(self):

        return self.state_action_dict
        