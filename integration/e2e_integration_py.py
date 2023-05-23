# %%
#this is end to end integration. processing steps as stated as a natural language. this is converted to actions

# %%
import pickle
instruction_dir = 'process_steps.pkl'

# %%
#use this cell to write some sample tsts for debugging:

""" 
process_steps = ["navigate to url:http://localhost:3000/webtrain1", "enter 'First name test' in field 'First Name'"]

#pre processing instructions: split in to instructions and data 

instruction_data = {}
for index in range(len(process_steps)):
    instruction_data[index] = {}
    if "navigate to url:" in process_steps[index]:
        instruction_data[index]['process'] = 'navigate to url'
        instruction_data[index]['data'] = process_steps[index].replace('navigate to url:',"")
    else:

        instruction_data[index]['process'] = 'enter data in'
        instruction_data[index]['data'] = {}
        

        pos = [pos for pos, char in enumerate(process_steps[index]) if char == "'"]
        instruction_data[index]['data']['field name'] = process_steps[index][pos[2]:pos[3]+1].strip("'")
        instruction_data[index]['data']['value'] = process_steps[index][pos[0]:pos[1]+1].strip("'")

#print(instruction_data)

with open(instruction_dir, 'wb') as f:
            pickle.dump(instruction_data, f)

"""
# %%
#view the output of test_setup
with open(instruction_dir, 'rb') as f:
    instruction_data = pickle.load(f)

# %%

##set system path to include relevent modules
import sys
import pathlib
import os
root_folder = pathlib.Path(os.getcwd()) #.parent.parent.resolve()
script_dir = os.path.join(root_folder, "DQN")
sys.path.append(os.path.dirname(script_dir))


# %%

#fetch action for each instruction
from DQN.dqn_execution_nlp import DQNExecution #import dqn module for prediction
from integration.nlp_to_action.env_setup_nlp import EnvNlp

#instantiate environment
output_sequence_length = 5
checkpoint_path = 'integration/nlp_to_action/checkpoint_nlp_action.pth'
vocab_data = ["navigate", "to", "click", "on", "enter","data","in","the","field"]

#nlp_action_dict

nlp_action_dict = { 
                    0: 
                        { 
                        "nl": "navigate to webpage",
                        "action": 0
                        },
                     1: 
                        { 
                        "nl": "click on button",
                        "action": 1
                        },
                     2:
                        { 
                        "nl": "enter data in",
                        "action": 2
                        }
                }

env_nlp = EnvNlp(training_data = "", vocabulary = vocab_data, max_tokens=100, output_sequence_length = output_sequence_length, nlp_action_dict = nlp_action_dict)




execute = DQNExecution(state_size = output_sequence_length, action_size = 3,env = env_nlp, checkpoint_path = checkpoint_path) #, num_states, num_vertical_grid, num_horizontal_grid, env2d, checkpoint_path)


def fetch_actions(instruction, env):
    state_arr = []
    state_arr.append(instruction)
    state = env.get_token(state_arr)
    action = execute.dqn_execute(state)

    return action


# %%
#location the input field

#instantiate environment
"""
Environment behaviour is defined
"""
from DQN.env_setup import EnvWeb2D
from DQN.dqn_execution import DQNExecution as DQNExecutionLocation

window_size = (1200,1000)
num_horizontal_grid = 10
num_vertical_grid = 10

grid = (num_vertical_grid, num_horizontal_grid)
num_states = 20 #this is based on 

dict_web_dir = 'integration/data/element_dictionary.pkl' #location of input and label fields and grid numbers stored as dictionary after processing
screenshots_dir = 'integration/screenshots/' #directory to help visulaization

env2d = EnvWeb2D(dict_web_dir, num_states,grid, window_size)

#define checkpoint for trining and testing
checkpoint_path = 'integration/checkpoint_label_input.pth'


execute_location = DQNExecutionLocation(num_states, num_vertical_grid, num_horizontal_grid, env2d, checkpoint_path)

dict_label_location =  env2d.get_data()

def fetch_location(execute_location,state, window_size):
    location = execute_location.dqn_execute(state/(max(window_size)))

    return location

# %%
#get state
import numpy as np
def get_state(label_index,webpage,state_size,dict_label_location):
    
    state = dict_label_location[webpage]['state'][label_index]

    if len(state)< state_size:
        for _ in range(0, state_size - len(state)):
            state = np.append(state, 0)

    return state #/max(window_size) #note this normalization is based on min = 0, and (x-x_min)/(x_max - x_min)

# %%
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

        

def get_label_index(webpage,label_name):       
    chrome_options = Options()
    chrome_options.headless = True
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(webpage)
    id_lst = []
    ids = driver.find_elements("xpath","//label")
    for id in ids:
        id_lst.append(id.text)

    driver.close()
    
    if label_name in id_lst:
        return id_lst.index(label_name)
    
    else:
        return "label not found!"
    
    

#label_index = get_label_index("http://localhost:3000/webtest1","First Name")
#print(label_index)

# %%
from navigation_web import NavigateWeb
import pickle


dict_web_dir = 'integration/data/element_dictionary.pkl' #location of input and label fields and grid numbers stored as dictionary after processing
screenshots_dir = 'integration/screenshots/' #directory to help visulaization
#view the output of test_setup

with open(dict_web_dir, 'rb') as f:
    dict_web_elements = pickle.load(f)

#print(dict_web_elements['http://localhost:3000/webtrain1']) 


# %%





# %%

import time
nav_web_test = NavigateWeb(window_size, grid)
web_page = ""
previous_web_page = ""
grid_num_lst = []
data_lst = []
count_webpage = 0


for item in instruction_data:
    instruction = instruction_data[item]['process']
    action = fetch_actions(instruction, env_nlp)

    if type(instruction_data[item]['data']) == dict:
        field_name = instruction_data[item]['data']['field name']
        field_value = instruction_data[item]['data']['value']
        label_index_n = get_label_index(web_page,field_name)
        state = get_state(label_index_n,web_page,num_states,dict_label_location)
        grid_num = fetch_location(execute_location,state, window_size)
        grid_num_lst.append(grid_num)
        data_lst.append(field_value)
     
    else:
        previous_web_page = web_page
        web_page = instruction_data[item]['data']
        count_webpage+=1

    if (action == 0 and count_webpage>=2):

        
        nav_web_test.main_data(previous_web_page, dict_web_elements[web_page], grid_num_lst, data_lst, screenshots_dir)
        grid_num_lst = []
        data_lst = []
        time.sleep(2)

    
nav_web_test.main_data(web_page, dict_web_elements[web_page], grid_num_lst, data_lst, screenshots_dir)
time.sleep(5)        
nav_web_test.close_webpage()    


# %%








