import numpy as np
import copy
import MCTS
import Node
import BTAI_agent
import itertools
from pymdp.agent import Agent
from pymdp.utils import plot_beliefs, plot_likelihood
from pymdp import utils
from pymdp.maths import softmax
import time


class GridWorldEnv():

    def __init__(self, starting_state = (0,0)):

        self.init_state = starting_state
        self.current_state = self.init_state
        print(f'Satrting state is {starting_state}')

    def step(self, action_label):

        (Y, X) = self.current_state

        if action_label == "UP": 
            Y_new = Y - 1 if Y > 0 else Y
            X_new = X
        
        elif action_label == "DOWN":
            Y_new = Y + 1 if Y < 2 else Y
            X_new = X

        elif action_label == "LEFT":
            Y_new = Y
            X_new = X - 1 if X > 0 else X
        
        elif action_label == "RIGHT":
            Y_new = Y
            X_new = X + 1 if X < 2 else X

        elif action_label == "STAY":
            Y_new, X_new = Y, X

        self.current_state = (Y_new, X_new) # store the new location
        obs = self.current_state # agent always observes the locations they are in

        return obs
    
    def reset(self):
        self.current_state = self.init_state
        print(f'Re-initialized to location {self.init_state}')
        obs = self.current_state
        print(f'..and sampled observation {obs}')

        return obs

def create_B_matrix():
    B = np.zeros((len(grid_locations), len(grid_locations), len(actions)))

    for action_id, action_label in enumerate(actions):
      for curr_state, grid_location in enumerate(grid_locations):
        y, x = grid_location
        #x, y  = grid_location
        if action_label == "UP":
            next_y = y - 1 if y > 0 else y 
            next_x = x
        elif action_label == "DOWN":
            next_y = y + 1 if y < 2 else y 
            next_x = x
        elif action_label == "LEFT":
          next_x = x - 1 if x > 0 else x 
          next_y = y
        elif action_label == "RIGHT":
          next_x = x + 1 if x < 2 else x 
          next_y = y
        elif action_label == "STAY":
          next_x = x
          next_y = y
        new_location = (next_y, next_x)
        next_state = grid_locations.index(new_location)
        B[next_state, curr_state, action_id] = 1.0
    return B
# Set up
grid_locations = list(itertools.product(range(3), repeat = 2)) # 3x3 grid
n_states = len(grid_locations) # 9 states: [0, 8]
n_observations = len(grid_locations) # 9 observations: [0, 8]
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"] # each action is indexized from 0 to 4

# The generative model
A = np.eye(n_observations, n_states) # likelihood
B = create_B_matrix() # transition matrix 
C = utils.onehot(grid_locations.index((2,2)), n_observations) # Preference
D = utils.onehot(grid_locations.index((0,0)), n_states) # Prior over initial state

trail_prior = np.array([[0,1, 1, 1, 2, 3, 2, 3, 10]])
trail_prior = utils.norm_dist_obj_arr(trail_prior)

env = GridWorldEnv(starting_state=(0,0))

agent_info= {'A' : A, 
             #'pA' : pA, 
             #'pB':pB, 
             'B' : B, 
             #'control_fac_idx':controllable_indeces,
             #'modalities_to_learn':learnable_modalities,
             #'lr_pA':0.25,
             #'lr_pB':0.25,
             'use_param_info_gain':True,
             'use_states_info_gain':True}

max_planning_iteration = 10
agent = BTAI_agent.BTAI_agent(MCTS.MCTS(2.4), 
                              gamma = 1.0, 
                              max_planning_iteration = 70, 
                              action_selection="SAMPLE",
                              **agent_info)

agent.C = utils.onehot(grid_locations.index((2,2)), n_observations)
#agent.C = trail_prior[0]
agent.D = utils.onehot(grid_locations.index((0,0)), n_states)



T = 30

obs = env.reset()

action_comb = np.arange(5)
action_comb = [[item] for item in np.nditer(action_comb)]

start_time = time.process_time()

for t in range(T):
    
    obs_idx = grid_locations.index(obs)
    print(f'Time: {t} : obs = {obs_idx}')
    qx = agent.infer_states([obs_idx])
    root = Node.Node(state_posterior = qx, n_actions = action_comb, obs_prior = [agent.C])
    action = agent.step(root)
    action = actions[action[0]]

    obs = env.step(action)

end_time = time.process_time()
print(f'*****CPU Exectution time : {end_time - start_time}*****')
agent.affective_response()