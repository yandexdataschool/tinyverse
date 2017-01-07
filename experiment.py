"""a minimalistic experiment designed to test the framework"""

import numpy as np
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from agentnet.resolver import EpsilonGreedyResolver
from agentnet import Agent
import gym
from player import save_all_params

class Experiment:
    def __init__(self,
                 game="CartPole-v0",
                 params_name = 'dummy.weights',
                 n_observations = 4,
                 n_actions = 2):
        """a bloody-stupid experiment setup that works with toy problems like cartpole or lunarlander"""
        
        self.game = game
        self.params_name = params_name
        self.sequence_length = 10
        
        #observation
        inp = InputLayer((None,n_observations),)

        #network body
        h1 = DenseLayer(inp,100,nonlinearity=sigmoid)
        h2 = DenseLayer(h1,100,nonlinearity=sigmoid)

        #a layer that predicts Qvalues
        qvalues_layer = DenseLayer(h2,n_actions,nonlinearity=lambda x:x)

        #To pick actions, we use an epsilon-greedy resolver (epsilon is a property)
        from agentnet.resolver import EpsilonGreedyResolver
        action_layer = EpsilonGreedyResolver(qvalues_layer)
        action_layer.epsilon.set_value(np.float32(0.05))
        
        self.agent = Agent(observation_layers=inp,action_layers=action_layer)

        save_all_params(self.agent, self.params_name)


    def make_env(self):
        """spawn a new environment instance"""
        return gym.make(self.game)

