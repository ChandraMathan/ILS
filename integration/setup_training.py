""" 
Setting up the data for training

"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import math

import pickle
import time
import os

class TrainingSetup:
    """ 
    this class outtputs a dictionary for training a DQN agent

    """
    def __init__(self, webpages, window_size, grid, num_states):
        
        """ 
        Params
        ======
            webpages (list): list of webpages for training
            window_size (tuple): size of webpage in (height, width)
            grid (tuple): number of verticle and horizontal grids in (vertical_grid, horizontal_grid)
            num_states (int): number of states 
        """

        self.webpages = webpages
        self.window_size = window_size
        self.grid = grid
        self.num_states = num_states
        
        self.grid_vertical = self.grid[0]
        self.grid_horizontal = self.grid[1]

        
        self.chrome_options = Options()
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.driver.set_window_size(self.window_size[0],self.window_size[1])
        
        self.element_dict = {}

    def open_webpage(self, webpage):

        self.driver.get(webpage)

    def close_webpage(self):

        self.driver.close()

    def get_ids(self):

        """ gets all the ids in the webpage """
        id_lst = []
        ids = self.driver.find_elements("xpath","//*[@id]")
        for id in ids:
            id_lst.append(id.get_attribute('id'))

        if 'root' in id_lst:
            index = id_lst.index("root")
            id_lst.pop(index)


        return id_lst
    
    def html_size(self):
        """ returns the size of webpage """

        element_html = self.driver.find_element(By.XPATH, '/html')

        image_width = element_html.size['width']
        image_height = element_html.size['height']

        return image_width, image_height

    def update_location(self, webpage,ids):
        """ updates the location of the elements"""

        self.element_dict[webpage] = {}

        for id in ids:

            
            self.element_dict[webpage][id] = {}
            self.element_dict[webpage][id]['label'] = {}
            self.element_dict[webpage][id]['input'] = {}
            element_label = self.driver.find_element("xpath","//label[@for='{}']".format(str(id)))
            element_input = self.driver.find_element("xpath","//*[@id='{}']".format(str(id)))
            
            self.element_dict[webpage][id]['label']['x_location'] = element_label.location['x']
            self.element_dict[webpage][id]['label']['y_location'] = element_label.location['y']
            self.element_dict[webpage][id]['label']['width'] = element_label.size['width']
            self.element_dict[webpage][id]['label']['height'] = element_label.size['height']
            self.element_dict[webpage][id]['label']['x_centre'] = element_label.location['x'] + round((element_label.size['width']/2.0),2)
            self.element_dict[webpage][id]['label']['y_centre'] = element_label.location['y'] + round((element_label.size['height']/2.0),2)
                                                                                                
            self.element_dict[webpage][id]['input']['x_location'] = element_input.location['x']
            self.element_dict[webpage][id]['input']['y_location'] = element_input.location['y']
            self.element_dict[webpage][id]['input']['width'] = element_input.size['width']
            self.element_dict[webpage][id]['input']['height'] = element_input.size['height']
            self.element_dict[webpage][id]['input']['x_centre'] = element_input.location['x'] + round((element_input.size['width']/2.0),1)
            self.element_dict[webpage][id]['input']['y_centre'] = element_input.location['y'] + round((element_input.size['height']/2.0),1)

    
    def location_to_grid(self, webpage, ids, image_width, image_height):
        """ updates the dictionary with grid number based on the location of the element """

        for id in ids:

            
            self.element_dict[webpage][id]['label']['grid_num'] = {}
            self.element_dict[webpage][id]['input']['grid_num'] = {}

            y_centre_label = self.element_dict[webpage][id]['label']['y_centre']
            x_centre_label = self.element_dict[webpage][id]['label']['x_centre']

            y_centre_input = self.element_dict[webpage][id]['input']['y_centre']
            x_centre_input = self.element_dict[webpage][id]['input']['x_centre']
            

            row_num_label = math.floor(y_centre_label*self.grid_vertical/image_height) #this is the previous row of the element of interest
            col_num_label = math.ceil(x_centre_label*self.grid_horizontal/image_width) #this is the column of the element of interest
            grid_num_label = (row_num_label*self.grid_vertical) + col_num_label

            row_num_input = math.floor(y_centre_input*self.grid_vertical/image_height) #this is the previous row of the element of interest
            col_num_input = math.ceil(x_centre_input*self.grid_horizontal/image_width) #this is the column of the element of interest
            grid_num_input = (row_num_input*self.grid_vertical) + col_num_input
            
           

            self.element_dict[webpage][id]['label']['grid_num'] = grid_num_label
            self.element_dict[webpage][id]['input']['grid_num'] = grid_num_input

    def main(self):


        for webpage in self.webpages:
            
            time.sleep(2)
            self.open_webpage(webpage)
            time.sleep(2)
            ids = self.get_ids()
            
            image_width, image_height = self.html_size()
            self.update_location(webpage,ids)
            self.location_to_grid(webpage,ids,image_width, image_height)
            time.sleep(2)
        
        self.close_webpage()

        
        with open('integration/data/element_dictionary.pkl', 'wb') as f:
            pickle.dump(self.element_dict, f)



webpages = ["http://localhost:3000/web1","http://localhost:3000/web2"]
open_web = TrainingSetup(webpages,(1200,1000),(10,10),"")
open_web.main()

with open('integration/data/element_dictionary.pkl', 'rb') as f:
    element_sol = pickle.load(f)

print(element_sol)

