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

from dqn_agent import Agent

class Training:
    """
    train dqn
    """

    def __init__(self, state_size, num_vertical_grid, num_horizontal_grid, env):
        
        self.state_size = state_size
        self.num_vertical_grid = num_vertical_grid
        self.num_horizontal_grid = num_horizontal_grid
        self.env = env
        self.action_size = self.num_vertical_grid * self.num_horizontal_grid

        self.agent = Agent(state_size=state_size, action_size=self.action_size, seed=2)


    def dqn(self, n_episodes, max_t, eps_start, eps_end, eps_decay, score_length, terminal_score):
        
        """Deep Q-Learning.
        
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
            score_length (int): keep track of the score for the given length
            terminal_score (float): terminate the training when average score achieves the set score 
        """
        

        consequetive_episode  = []
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=score_length)  # last xxx scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            state = self.env.reset() #note state is atuple with a state 
            
            score = 0
            for t in range(max_t):
                action = self.agent.act(state, eps) #select normalized state
                next_state, reward,done = self.env.env_behaviour(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 5000 == 0:
                print('\n')
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)),eps)
            
            if i_episode % 1000 == 0:
                torch.save(self.agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                
            if np.mean(scores_window) >=terminal_score: #and len(consequetive_episode)>7: # and i_episode >= 1000:

                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(self.agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
                
        return scores
