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
    
        num_cols, num_rows = math.modf(grid_num / self.grid_horizontal)
        num_cols = round(num_cols,2)*10
        
        y_location = ((num_rows*self.image_height)/self.grid_horizontal) + (self.image_height*0.5/self.grid_horizontal)
        x_location = ((num_cols*self.image_width)/self.grid_vertical) - (self.image_width*0.5/self.grid_vertical)
        
        return (x_location, y_location)

    def nav_web(self, input_location):

        action = webdriver.common.action_chains.ActionChains(self.driver)
        x_offset = round(input_location[0] - self.image_width/2.0,1)
        y_offset = round(input_location[1] - self.image_height/2.0,1)
    
        action.move_to_element_with_offset(self.element_html, x_offset, y_offset).click().send_keys("yes").perform()


    def screenshots(self, screenshot_dir):

        current_time = str(datetime.datetime.now())
        char_remov = [":", " ", ".", "-"]
        
        for char in char_remov:
            current_time = current_time.replace(char,"")

        self.element_html.screenshot(screenshot_dir+"web_executed_"+current_time+".jpg")

    def main(self, webpage, input_grid, screenshot_dir):

        """
        sequence of actions for execution
        Params: 'input_locations': grid number for the input field
        """
        self.open_webpage(webpage)
        self.html_size()
        for item in input_grid:
            coordinates = self.grid_to_coordinates(item)
            self.nav_web(coordinates)
        
        self.screenshots(screenshot_dir)
        
