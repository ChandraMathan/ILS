
import random
import numpy as np
import tensorflow as tf


class EnvNlp:

    def __init__(self, training_data, vocabulary, max_tokens, output_sequence_length, nlp_action_dict):

        """
        Params: 
        
        training_data: is a list of input strings typically used
        nlp_action_dict: is a dictionary where input string and the corresponding action is specified
        
        """
        self.training_data = training_data
        self.vocabulary = vocabulary
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.nlp_action_dict = nlp_action_dict

        self.state_size = max_tokens

        # Create the layer, passing the vocab directly.
        self. vectorize_layer = tf.keras.layers.TextVectorization(
                                            max_tokens = self.max_tokens,
                                            output_mode = 'int',
                                            output_sequence_length = self.output_sequence_length,
                                            vocabulary = vocabulary
                                )

        # Create the model that uses the vectorize text layer
        self.model = tf.keras.models.Sequential()  

        #create input layer
        self.model.add(tf.keras.Input(shape=(1,), dtype=tf.string))

        # first layer in our model is the vectorization layer.
        self.model.add(self.vectorize_layer)

        self.state = None
        self.curr_data = None
        self.curr_action = None

    def reset(self):

        """ 
        
        """

        self.curr_data = random.choice(self.training_data)
        
        for key in self.nlp_action_dict:
            
            if self.nlp_action_dict[key]['nl'] == self.curr_data[0]:
                self.curr_action = self.nlp_action_dict[key]['action']
                self.state = np.asarray(self.model.predict(self.curr_data))[0]

       

        return self.state
    
    def env_behaviour(self, action):

        if self.curr_action == action:
            reward = 0.1
            done = True
            #next_state = np.full((self.state_size, ), -1.0)
            next_state = np.full((5, ), -1.0)
        
        else:
            reward = -0.1
            done = False
            next_state = self.state
            
        
        return next_state, reward, done

    def get_token(self, data):

        return np.asarray(self.model.predict(data))[0]
        
