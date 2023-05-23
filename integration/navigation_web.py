from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import datetime
import math
import time

class NavigateWeb:
    """ 
    navigate web 
    """

    def __init__(self, window_size, grid):
        
        """ 
        Params
        ======
            webpages (list): list of webpages for training
            window_size (tuple): size of webpage in (height, width)
            grid (tuple): number of verticle and horizontal grids in (vertical_grid, horizontal_grid)
            num_states (int): number of states 
        """

        
        self.window_size = window_size
        
        
        self.chrome_options = Options()
        self.chrome_options.add_experimental_option("detach", True)
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.driver.set_window_size(self.window_size[0],self.window_size[1])

        self.grid = grid
        
        
        self.grid_vertical = self.grid[0]
        self.grid_horizontal = self.grid[1]
        
        

    def open_webpage(self, webpage):

        self.driver.get(webpage)

    def close_webpage(self):

        self.driver.close()

    
    def html_size(self):
        """ returns the size of webpage """

        self.element_html = self.driver.find_element(By.XPATH, '/html')

        self.image_width = self.element_html.size['width']
        self.image_height = self.element_html.size['height']

    def grid_to_coordinates(self,grid_num):
        
        """ given the grid number, returns co-ordinates of middle of the grid """
        num_cols, num_rows = math.modf(grid_num / self.grid_horizontal)
        num_cols = round(num_cols,2)*10
        
        y_location = ((num_rows*self.image_height)/self.grid_horizontal) + (self.image_height*0.5/self.grid_horizontal)
        x_location = ((num_cols*self.image_width)/self.grid_vertical) - (self.image_width*0.5/self.grid_vertical)


        return (x_location, y_location)
    
    def coordinates_from_data(self, find_grid, location_data):

        """ 
        looks for the grid number defined in find_grid in location_data
            if not found, picks the closest grid and returns the coordinates 
        """ 
        grid_found = False
        
        for item in location_data:
            if location_data[item]['input']['grid_num'] == find_grid:
                x_location = location_data[item]['input']['x_centre']
                y_location = location_data[item]['input']['y_centre']
                grid_found = True
 

        
        if grid_found == False:
            grid_diff = {}
            for items in location_data:
                grid_diff[items] = abs(location_data[items]['input']['grid_num'] - find_grid)

            dict_min_key = min(grid_diff, key=grid_diff.get)
            x_location = location_data[dict_min_key]['input']['x_centre']
            y_location = location_data[dict_min_key]['input']['y_centre']


        return (x_location, y_location)

    def nav_web(self, input_location,input_value):

        time.sleep(1)
        action = webdriver.common.action_chains.ActionChains(self.driver)
        x_offset = round(input_location[0] - self.image_width/2.0,1)
        y_offset = round(input_location[1] - self.image_height/2.0,1)

        action.move_to_element_with_offset(self.element_html, x_offset, y_offset).click().send_keys(input_value).perform()


    def screenshots(self, screenshot_dir):

        current_time = str(datetime.datetime.now())
        char_remov = [":", " ", ".", "-"]
        
        for char in char_remov:
            current_time = current_time.replace(char,"")

        self.element_html.screenshot(screenshot_dir+"web_executed_"+current_time+".jpg")

    def main(self, webpage, webele_location ,input_grid, screenshot_dir):

        """
        sequence of actions for execution
        Params: 'input_locations': grid number for the input field
               'webelelocation': dictionary of location of web lements including label and input field 
        """
        self.open_webpage(webpage)
        self.html_size()
        for item in input_grid:
            
            coordinates = self.coordinates_from_data(item,webele_location)
            #coordinates = self.grid_to_coordinates(item)
            self.nav_web(coordinates,"yes")
        
        #self.screenshots(screenshot_dir)

    def main_data(self, webpage, webele_location ,input_grid, data, screenshot_dir):

        """
        sequence of actions for execution
        Params: 'input_locations': grid number for the input field
               'webelelocation': dictionary of location of web lements including label and input field 
        """
        self.open_webpage(webpage)
        self.html_size()
        for item in range(len(input_grid)):
            
            coordinates = self.coordinates_from_data(input_grid[item],webele_location)
            #coordinates = self.grid_to_coordinates(item)
            self.nav_web(coordinates,data[item])
            #self.nav_web(coordinates,"yes")
        
        self.screenshots(screenshot_dir)
        
