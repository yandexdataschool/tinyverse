"""a minimalistic experiment designed to test the framework"""

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *

from agentnet import Agent
from agentnet.resolver import EpsilonGreedyResolver
from agentnet.learning import qlearning

import gym



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
        
        self.agent = Agent(observation_layers=inp,
                           policy_estimators=qvalues_layer,
                           action_layers=action_layer)


    def make_env(self):
        """spawn a new environment instance"""
        return gym.make(self.game)
    
    def build_train_step(self,replay,inputs=()):
        """Compiles a function to train for one step"""
        _,_,_,_,qvalues_seq = self.agent.get_sessions(
            replay,
            session_length=self.sequence_length,
            experience_replay=True,
        )
        
        #get reference Qvalues according to Qlearning algorithm


        elwise_mse_loss = qlearning.get_elementwise_objective(qvalues_seq,
                                                              replay.actions[0],
                                                              replay.rewards,
                                                              replay.is_alive,
                                                              gamma_or_gammas=0.99,)

        #compute mean over "alive" fragments
        loss = elwise_mse_loss.sum() / replay.is_alive.sum()
        
        
        weights = get_all_params(self.agent.action_layers+list(self.agent.agent_states))
        
        # Compute weight updates
        updates = lasagne.updates.rmsprop(loss,weights,learning_rate=0.01)
        
        #compile train function
        return theano.function(inputs,loss,updates=updates,allow_input_downcast=True)
