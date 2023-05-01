import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import pickle

from dqn_agent_execution import Agent

class DQNExecution:

    """
    Execute based on trained parameters for DQN
    """

    def __init__(self, state_size, num_vertical_grid, num_horizontal_grid, env):
        
        self.state_size = state_size
        self.num_vertical_grid = num_vertical_grid
        self.num_horizontal_grid = num_horizontal_grid
        self.env = env
        self.action_size = self.num_vertical_grid * self.num_horizontal_grid

        self.agent = Agent(state_size=state_size, action_size=self.action_size, seed=2)

    def dqn_execute(self, state):

        eps = 0.0

        action = self.agent.act(state, eps)

            
                
        return action

