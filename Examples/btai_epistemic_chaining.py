import numpy as np

from pymdp import utils

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

import MCTS
import Node
import BTAI_agent

from itertools import product
from pymdp import utils

import time

# Defining the environment
grid_dims = [5, 7]
num_grid_points = np.prod(grid_dims)

grid = np.arange(num_grid_points).reshape(grid_dims)
it = np.nditer(grid, flags=["multi_index"])

loc_list = []
while not it.finished:
    loc_list.append(it.multi_index)
    it.iternext()

cue1_location = (2, 0)
cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]

# names of the reward conditions and their locations
reward_conditions = ["TOP", "BOTTOM"]
reward_locations = [(1, 5), (3, 5)]

# Visualize the world grid - environment
fig, ax = plt.subplots(figsize = (10,6))

X, Y = np.meshgrid(np.arange(grid_dims[1]+1), np.arange(grid_dims[0]+1))
h = ax.pcolormesh(X, Y, np.ones(grid_dims), edgecolors='k', vmin = 0, vmax = 30, linewidth=3, cmap = 'coolwarm')
ax.invert_yaxis()

# Put gray boxes around the possible reward locations
reward_top = ax.add_patch(patches.Rectangle((reward_locations[0][1],reward_locations[0][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor=[0.5, 0.5, 0.5]))
reward_bottom = ax.add_patch(patches.Rectangle((reward_locations[1][1],reward_locations[1][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor=[0.5, 0.5, 0.5]))

text_offsets = [0.4, 0.6]

cue_grid = np.ones(grid_dims)
cue_grid[cue1_location[0],cue1_location[1]] = 15.0
for ii, loc_ii in enumerate(cue2_locations):
  row_coord, column_coord = loc_ii
  cue_grid[row_coord, column_coord] = 5.0
  ax.text(column_coord+text_offsets[0], row_coord+text_offsets[1], cue2_loc_names[ii], fontsize = 15, color='k')
h.set_array(cue_grid.ravel())

# The Generative model
"""
Hidden states factors:
 1. Location: as many levels as there are grid locations
 2. Cue 2: 4 levels, position of cue 2 - where the cue_2 is (cue_1 is fixed)
 3. Reward condition: 2 levels - where the reward is

Observation modalities:
 1. Location
 2. Cue 1: 5 levels = null + the 4 possible locations of cue_2 
 3. Cue 2: 3 levels = null + the 2 reward possiblities 
 4. Reward: 3 levels = reward, loss or null
"""

num_states = [num_grid_points, len(cue2_locations), len(reward_locations)]

cue1_names = ['Null'] + cue2_loc_names
cue2_names = ['Null', 'reward_on_top', 'reward_on_bottom']
reward_names = ['Null', 'Cheese', 'Shock']

num_obs = [num_grid_points, len(cue1_names), len(cue2_names), len(reward_names)]

# Likelihood A
A_m_shapes = [ [o_dim] + num_states for o_dim in num_obs] 
A = utils.obj_array_zeros(A_m_shapes)

# A modlaity 1
A[0] = np.tile(np.expand_dims(np.eye(num_grid_points), (-2, -1)), (1, 1, num_states[1], num_states[2]))
# A modality 2
A[1][0,:,:,:] = 1.0 # makes Null the most likely observation everywhere: P(o{cue1} = Null | S{location, cue2, reward_condition})

for i, cue_loc2_i  in enumerate(cue2_locations):# i = ith cue index, cue2_loc_i = lcoation of ith cue
   A[1][0,loc_list.index(cue1_location),i,:] = 0.0 # P(o{cue1} = null | S{loc = cue, cue2 = i, reward_condition = any})
   A[1][i+1,loc_list.index(cue1_location),i,:] = 1.0 

# A modality 3
A[2][0,:,:,:] = 1.0
for i, cue_loc2_i in enumerate(cue2_locations): 
   A[2][0,loc_list.index(cue_loc2_i),i,:] = 0.0
   A[2][1,loc_list.index(cue_loc2_i),i,0] = 1.0 
   A[2][2,loc_list.index(cue_loc2_i),i,1] = 1.0 

# A modality 4
A[3][0,:,:,:] = 1.0

rew_top_idx = loc_list.index(reward_locations[0])
rew_bott_idx = loc_list.index(reward_locations[1])

# fill out the contingencies when the agent is in the "TOP" reward location
A[3][0,rew_top_idx,:,:] = 0.0
A[3][1,rew_top_idx,:,0] = 1.0 # P(o{reward = cheese} | S{loc = top_reward, cue2 = any, reward_condition = top}) = 1.0
A[3][2,rew_top_idx,:,1] = 1.0 # P(o{reward = shock} | S{loc=top_reward, cue2 = any, reward_condition = bottom}) = 1.0

# fill out the contingencies when the agent is in the "BOTTOM" reward location
A[3][0,rew_bott_idx,:,:] = 0.0
A[3][1,rew_bott_idx,:,1] = 1.0
A[3][2,rew_bott_idx,:,0] = 1.0

# The transition model B
# One per each hidden states factor

num_controls = [5, 1, 1] # number of action per each state: 5 actions in location state, other 2 are not influenced by the agent
B_f_shapes = [ [ns, ns, num_controls[f]] for f, ns in enumerate(num_states)]
B = utils.obj_array_zeros(B_f_shapes)

actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

# B[0]

for action_id, action_label in enumerate(actions):
   for curr_state, grid_location in enumerate(loc_list):
        y, x = grid_location # just a convention: y - horizontal, x - vertical

        if action_label == "UP":
            next_y = y - 1 if y > 0 else y 
            next_x = x
        elif action_label == "DOWN":
            next_y = y + 1 if y < (grid_dims[0]-1) else y 
            next_x = x
        elif action_label == "LEFT":
            next_x = x - 1 if x > 0 else x 
            next_y = y
        elif action_label == "RIGHT":
            next_x = x + 1 if x < (grid_dims[1]-1) else x 
            next_y = y
        elif action_label == "STAY":
            next_x = x
            next_y = y

        new_location = (next_y, next_x)
        next_state = loc_list.index(new_location)
        B[0][next_state, curr_state, action_id] = 1.0 

# B[1], B[2]
B[1][:,:,0] = np.eye(num_states[1]) # state doesn't change
B[2][:,:,0] = np.eye(num_states[2]) # state doesn't change

# C vector
C = utils.obj_array_zeros(num_obs)

C[3][1] = 1.0 # prior on "cheese"
C[3][2] = 0.0 # prior on "shock"
#C[1][1:] = [1e-7]*4
#C[2][1:] = [1e-7]*2
#C[0][:] = [1e-7]*len(C[0])

# D vector
D = utils.obj_array_uniform(num_states)
D[0] = utils.onehot(loc_list.index((0,0)), num_grid_points) # agents knows its starting loation 

# Generative Process: how the environment behaves
class GridWorldEnv():

    def __init__(self,starting_loc = (0,0), cue1_loc = (2, 0), cue2 = 'L1', reward_condition = 'BOTTOM'):
        self.init_loc = starting_loc
        self.current_location = self.init_loc

        self.cue1_loc = cue1_loc
        self.cue2_name = cue2
        self.cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
        self.cue2_loc = cue2_locations[self.cue2_loc_names.index(self.cue2_name)]

        self.reward_condition = reward_condition
        print(f'Starting location is {self.init_loc}, \n Reward condition is {self.reward_condition}, \n Cue is located in {self.cue2_name}')

    def step(self, action_label):
        (Y, X) = self.current_location

        if action_label == "UP": 
          
          Y_new = Y - 1 if Y > 0 else Y
          X_new = X

        elif action_label == "DOWN": 

          Y_new = Y + 1 if Y < (grid_dims[0]-1) else Y
          X_new = X

        elif action_label == "LEFT": 
          Y_new = Y
          X_new = X - 1 if X > 0 else X

        elif action_label == "RIGHT": 
          Y_new = Y
          X_new = X +1 if X < (grid_dims[1]-1) else X

        elif action_label == "STAY":
          Y_new, X_new = Y, X 

        self.current_location = (Y_new, X_new)
        loc_obs = self.current_location

        if self.current_location == self.cue1_loc:
           cue1_obs = self.cue2_name
        else:
           cue1_obs = 'Null'

        if self.current_location == self.cue2_loc:
           cue2_obs = cue2_names[reward_conditions.index(self.reward_condition)+1]
        else:
           cue2_obs = 'Null'
        
        #### NB
        if self.current_location == reward_locations[0]:
            if self.reward_condition == 'TOP':
               reward_obs = 'Cheese'
            else: 
               reward_obs = 'Shock'
        elif self.current_location == reward_locations[1]:
            if self.reward_condition == 'BOTTOM':
               reward_obs = 'Cheese'
            else:
               reward_obs = 'Shock'
        else:
            reward_obs = 'Null'

        return loc_obs, cue1_obs, cue2_obs, reward_obs
    
    def reset(self):
       self.current_location = self.init_loc
       print(f'Re-initialized location to {self.init_loc}')
       loc_obs = self.current_location
       cue1_obs = 'Null'
       cue2_obs = 'Null'
       reward_obs = 'Null'

       return loc_obs, cue1_obs, cue2_obs, reward_obs


# Building the agent

agent_info = {'A' : A, 
             #'pA' : pA, 
             #'pB':pB, 
             'B' : B, 
             'C' : C,
             'D' : D,
             'use_param_info_gain':True,
             'use_states_info_gain':True}

my_agent = BTAI_agent.BTAI_agent(MCTS.MCTS(2.4), 
                              gamma = 1.0, 
                              max_planning_iteration = 60, 
                              action_selection="SAMPLE",
                              **agent_info)

my_env = GridWorldEnv(starting_loc = (0,0), cue1_loc=(2,0), cue2='L4', reward_condition='TOP')
loc_obs, cue1_obs, cue2_obs, reward_obs = my_env.reset()

# Active Inference Loop
history_of_locs = [loc_obs]
obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs), reward_names.index(reward_obs)]


n_controls = my_agent.num_controls
n_controls_extended = [np.arange(item) for item in n_controls]

if (len(my_agent.num_controls) > 1): 
    #actions = list(product(np.arange(agent.num_controls[0]), np.arange(agent.num_controls[1])))
    my_actions = list(product(*n_controls_extended))
    my_actions = [list(el) for el in my_actions]
else:
    my_actions = np.arange(my_agent.num_controls)
    my_actions = [[item] for item in np.nditer(my_actions)]
start = time.process_time()

T = 20 # timesteps

for t in range(T):
   
  qs = my_agent.infer_states(obs) # P(S | O), S ={loc, cue2, rew}, O={loc, cue1, cue2, rew}
  print(f'qs at time {t}: {qs}')
  root = Node.Node(state_posterior = qs, n_actions = my_actions, obs_prior = my_agent.C )
  chosen_action_id = my_agent.step(root) 

  movement_id = int(chosen_action_id[0])
  choice_action = actions[movement_id]

  print(f'Action at time {t}: {choice_action}')

  loc_obs, cue1_obs, cue2_obs, reward_obs = my_env.step(choice_action)
  
  obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs), reward_names.index(reward_obs)]
  print(f'Observation: {loc_obs, cue1_obs, cue2_obs, reward_obs}')
  history_of_locs.append(loc_obs)

  print(f'Grid location at time {t}: {loc_obs}')
  print(f'Reward at time {t}: {reward_obs}')

end = time.process_time()
print(f'*****CPU Exectution time : {end - start}*****')
print(my_agent.gamma_history)

import matplotlib.pyplot as plt
#plt.plot(my_agent.gamma_history)

# Visualization

all_locations = np.vstack(history_of_locs).astype(float) # create a matrix containing the agent's Y/X locations over time (each coordinate in one row of the matrix)

fig, ax = plt.subplots(figsize=(10, 6)) 

# create the grid visualization
X, Y = np.meshgrid(np.arange(grid_dims[1]+1), np.arange(grid_dims[0]+1))
h = ax.pcolormesh(X, Y, np.ones(grid_dims), edgecolors='k', vmin = 0, vmax = 30, linewidth=3, cmap = 'coolwarm')
ax.invert_yaxis()

# get generative process global parameters (the locations of the Cues, the reward condition, etc.)
cue1_loc, cue2_loc, reward_condition = my_env.cue1_loc, my_env.cue2_loc, my_env.reward_condition
reward_top = ax.add_patch(patches.Rectangle((reward_locations[0][1],reward_locations[0][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor='none'))
reward_bottom = ax.add_patch(patches.Rectangle((reward_locations[1][1],reward_locations[1][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor='none'))
reward_loc = reward_locations[0] if reward_condition == "TOP" else reward_locations[1]

if reward_condition == "TOP":
    reward_top.set_edgecolor('g')
    reward_top.set_facecolor('g')
    reward_bottom.set_edgecolor([0.7, 0.2, 0.2])
    reward_bottom.set_facecolor([0.7, 0.2, 0.2])
elif reward_condition == "BOTTOM":
    reward_bottom.set_edgecolor('g')
    reward_bottom.set_facecolor('g')
    reward_top.set_edgecolor([0.7, 0.2, 0.2])
    reward_top.set_facecolor([0.7, 0.2, 0.2])
reward_top.set_zorder(1)
reward_bottom.set_zorder(1)

text_offsets = [0.4, 0.6]
cue_grid = np.ones(grid_dims)
cue_grid[cue1_loc[0],cue1_loc[1]] = 15.0
for ii, loc_ii in enumerate(cue2_locations):
  row_coord, column_coord = loc_ii
  cue_grid[row_coord, column_coord] = 5.0
  ax.text(column_coord+text_offsets[0], row_coord+text_offsets[1], cue2_loc_names[ii], fontsize = 15, color='k')
  
h.set_array(cue_grid.ravel())

cue1_rect = ax.add_patch(patches.Rectangle((cue1_loc[1],cue1_loc[0]),1.0,1.0,linewidth=8,edgecolor=[0.5, 0.2, 0.7],facecolor='none'))
cue2_rect = ax.add_patch(patches.Rectangle((cue2_loc[1],cue2_loc[0]),1.0,1.0,linewidth=8,edgecolor=[0.5, 0.2, 0.7],facecolor='none'))

ax.plot(all_locations[:,1]+0.5,all_locations[:,0]+0.5, 'r', zorder = 2)

temporal_colormap = cm.hot(np.linspace(0,1,T+1))
dots = ax.scatter(all_locations[:,1]+0.5,all_locations[:,0]+0.5, 450, c = temporal_colormap, zorder=3)

ax.set_title(f"Cue 1 located at {cue1_loc}, Cue 2 located at {cue2_loc}, Cheese on {reward_condition}", fontsize=16)



my_agent.affective_response()

































