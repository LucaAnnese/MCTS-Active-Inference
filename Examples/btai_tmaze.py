import numpy as np
import copy
import MCTS
import Node
import BTAI_agent
import time

from itertools import product
from pymdp.agent import Agent
from pymdp.utils import plot_beliefs, plot_likelihood
from pymdp import utils
from pymdp.envs import TMazeEnvNullOutcome
from pymdp.maths import softmax

reward_conditions = ["Right Arm Better", "Left arm Better"]
location_observations = ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']
reward_observations = ['No reward','Reward!','Loss!']
cue_observations = ['Null','Cue Right','Cue Left']

reward_probabilities = [0.85, 0.15]
env = TMazeEnvNullOutcome(reward_probs=reward_probabilities)

A_gp = env.get_likelihood_dist()
B_gp = env.get_transition_dist()
A_gm, B_gm_noise = copy.deepcopy(A_gp), copy.deepcopy(B_gp)


# pA = utils.dirichlet_like(A_gp, scale = 1e16)
# pA[1][1:,1:3,:] = 1.0
# A_gm = utils.norm_dist_obj_arr(pA)
# A_gm[1] = softmax(A_gm[1] + abs(np.random.normal(0, 0.2,(3,4,2))))



# B_gm = copy.d
# eepcopy(B_gp)
# random_noise = abs(np.random.normal(0, 0.5, (4,4,4)))
# B_gm_noise = copy.deepcopy(B_gm)
# pB = utils.dirichlet_like(B_gm_noise, scale=1e16)
# for i in range(4):
#     pB[0][i,:,i] = 1
# B_gm_noise = utils.norm_dist_obj_arr(pB) 
# B_gm_noise[0] = softmax(B_gm_noise[0] + random_noise)


controllable_indeces = [0] # 
learnable_modalities = [1] # only the reward

# define the agent
agent_info= {'A' : A_gm, 
             #'pA' : pA, 
             #'pB':pB, 
             'B' : B_gm_noise, 
             #'control_fac_idx':controllable_indeces,
             #'modalities_to_learn':learnable_modalities,
             #'lr_pA':0.25,
             #'lr_pB':0.25,
             'use_param_info_gain':True,
             'use_states_info_gain':True}

max_planning_iteration = 10
agent = BTAI_agent.BTAI_agent(MCTS.MCTS(2.4), 
                              gamma = 0.5, 
                              max_planning_iteration = 50, 
                              action_selection="SAMPLE",
                              **agent_info)
context = ["Right", "Left"]


agent.D[0] = utils.onehot(0, agent.num_states[0])
agent.C[1][1] = 1.0
agent.C[1][2] = 0.0 # cannot be negative due to log computations
agent.C[2][1] = 1e-7
agent.C[2][2] = 1e-7
#print(f'Agent reward prior: {softmax(agent.C[1])}')
#print(f'Agent cue prior: {softmax(agent.C[2])}')
T = 15


obs = env.reset()                                      

actions = agent.expand_actions()
print(f'obs_prior : {agent.C}')
start_time = time.process_time()
for t in range(T):
    print(f'Time: {t} : obs = {obs}')
    qx = agent.infer_states(obs)
    print(f'Belief at time {t}: {qx}')
    #plot_beliefs(qx[1], title=f"Reward setting: {context[env.reward_condition]}")
    root = Node.Node(state_posterior = qx, n_actions = actions, obs_prior = agent.C )
    action = agent.step(root) 
    obs = env.step(action)
    msg = """[Step {}] Observation: [{}, {}, {}]"""
    print(msg.format(t, location_observations[int(obs[0])], reward_observations[int(obs[1])], cue_observations[int(obs[2])]))


end_time = time.process_time()
print(f'*****CPU Exectution time : {end_time - start_time}*****')
#plot_beliefs(qx[1], title=f"Reward setting: {context[env.reward_condition]}")
agent.affective_response()