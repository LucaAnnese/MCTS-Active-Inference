import Node
import pymdp.agent as Agent
from pymdp.utils import sample
from pymdp.maths import softmax
import numpy as np
from random import choice
from itertools import product
from scipy.stats import entropy

import matplotlib.pyplot as plt

class BTAI_agent(Agent.Agent):

    def __init__(self, MCTS, gamma = 1, max_planning_iteration = 10, action_selection = "MOST_VISITS", **kwargs ):
        super().__init__(**kwargs)
        self.mcts = MCTS
        self.generative_model = [] 
        self.mpi = max_planning_iteration
        self.gamma = gamma
        self.gamma_history = [gamma]
        self.energy_history = []
        self.action_selection = action_selection


        # n_factors = self.A[0].shape[1:]
        # #n_factors = [len(item) for item in self.state_posterior]
        # s_factors = [np.arange(el) for el in n_factors]
        # all_states = list(product(*s_factors))
        # self.all_states = all_states

    # override agent's method
    def sample_action(self, root): 
        children_visists = np.array([child.visits for child in root.children])
        children_costs = np.array([child.cost for child in root.children])

        if (self.action_selection == "MOST_VISITS"):
            most_visited_child = max(root.children, key=lambda x: x.visits)
            max_indices = np.where(children_visists == most_visited_child.visits)
            # If multiple nodes -> pick at random one of them
            max_indices = max_indices[0].tolist()
            if (len(max_indices) > 1):
                action = (root.children[choice(max_indices)]).in_action
            else:
                action = most_visited_child.in_action
        if (self.action_selection == "SAMPLE"):
            children_costs = np.array([child.cost for child in root.children])
            #children_visists = np.array([child.visits for child in root.children])
            G = softmax(-self.gamma * children_costs/children_visists)
            selected_node = sample(G)

            # Gamma computation
            self.gamma = 1 - entropy(G)
            self.gamma_history.append(self.gamma)
            # Epistemic/motivational increase

     
            action = root.children[selected_node].in_action
            #action = np.argmax(G)
            #action = np.argmax(children_visists)

        self.action = np.array(action)
        self.step_time()
    
        return action
    
    
    def expand_actions(self):
        n_controls = self.num_controls
        n_controls_extended = [np.arange(item) for item in n_controls]

        if (len(n_controls) > 1):
            actions = list(product(*n_controls_extended))
            actions = [list(el) for el in actions]
        else:
            actions = np.arange(n_controls)
            actions = [[item] for item in np.nditer(actions)]

        return actions
    
    #perform action-perception cycle
    def step(self, root): 

        '''
        Perform the action-perception cycle
        '''
        #print("Starting Branching Time Planning Simulation...")
        for t in range(self.mpi):
            #print(f'Simulation number: {t}')
            node = self.mcts.select_node(root) 
            #print("...node selected")
            e_nodes = self.mcts.expansion(node, self.A, self.B)
            #print("...selected node expanded correctly")
            self.mcts.evaluation(e_nodes, self.A)#, self.modalities_to_learn
            #print("...expansion evaluated")
            self.mcts.propagation(e_nodes, self.A)
            #print("...cost backpropagated")

        self.energy_history.append(root.cost)
        action = self.sample_action(root)
        
        return action
    
    def affective_response(self):
        plt.figure()
        plt.plot(np.arange(len(self.gamma_history)),self.gamma_history, '-o')
        plt.title("Affective response - Precision of beliefs about policies")
        plt.xlabel("Time steps")
        plt.ylabel("Gamma")

        plt.figure()
        plt.plot(np.arange(len(self.energy_history)), self.energy_history)

