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
from agentnet.memory import GRUCell

import gym


class Experiment:
    def __init__(self,
                 game="PongDeterministic-v0",
                 params_name = 'pong.weights',):
        """a bloody-stupid experiment setup that works with toy problems like cartpole or lunarlander"""
        
        self.game = game
        self.params_name = params_name
        self.sequence_length = 10
        
        observation_shape = (1,42,42)#same as env.observation_space.shape
        n_actions = 6 # same as env.action_space.n
        
        #observation
        inp = InputLayer((None,)+observation_shape,)
        
        #network body
        conv0 = Conv2DLayer(inp,32,3,stride=2,nonlinearity=elu)
        conv1 = Conv2DLayer(conv0,32,3,stride=2,nonlinearity=elu)
        conv2 = Conv2DLayer(conv1,64,3,stride=2,nonlinearity=elu)
        
        prev_gru = InputLayer((None,256))
        new_gru = GRUCell(prev_gru,flatten(conv2))
        
        dense1 = DenseLayer(new_gru,256,nonlinearity=tanh)

        #a layer that predicts Qvalues
        qvalues_layer = DenseLayer(dense1,n_actions,nonlinearity=None)

        #To pick actions, we use an epsilon-greedy resolver (epsilon is a property)
        from agentnet.resolver import EpsilonGreedyResolver
        action_layer = EpsilonGreedyResolver(qvalues_layer)
        action_layer.epsilon.set_value(np.float32(0.05))
        
        self.agent = Agent(observation_layers=inp,
                           policy_estimators=qvalues_layer,
                           agent_states={new_gru:prev_gru},
                           action_layers=action_layer)


    def make_env(self):
        """spawn a new environment instance"""
        env = gym.make(self.game)
        env = AtariRescale42x42(env)
        return env
    
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

    
    
    
###Some helper functions from 
###https://github.com/openai/universe-starter-agent/blob/master/envs.py
import cv2
def _process_frame42(frame):
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [1, 42, 42])
    return frame

from gym.core import ObservationWrapper
from gym.spaces.box import Box

class AtariRescale42x42(ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1,42, 42])

    def _observation(self, observation):
        return _process_frame42(observation)
