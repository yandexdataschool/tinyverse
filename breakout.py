"""a minimalistic experiment designed to test the framework"""

import gym
import lasagne
import numpy as np
import theano
import theano.tensor as T
from agentnet import Agent
from agentnet.environment import SessionBatchEnvironment
from agentnet.learning import a2c_n_step
from agentnet.memory import GRUCell
from lasagne.layers import *
from lasagne.nonlinearities import *
from tinyverse import Experiment, lazy

def make_experiment(db):
    """
    This is what's going to be created on "python tinyverse breakout.py ..."
    """
    return Breakout(db)


class Breakout(Experiment):
    def __init__(self,
                 db, #database instance (mandatory parameter)
                 sequence_length=20,  # how many steps to make before updating weights
                 game="BreakoutDeterministic-v0", #which game to play (uses gym.make)
                 ):
        """a simple experiment setup that works with atari breakout"""
        self.game=game
        super(Breakout, self).__init__(db, self.make_agent(), sequence_length=sequence_length)


    def make_env(self):
        """spawn a new environment instance"""
        env = gym.make(self.game)
        env = AtariRescale42x42(env)
        return env

    def make_agent(self,
                   observation_shape=(1, 42, 42), # same as env.observation_space.shape
                   n_actions = 6,  # same as env.action_space.n
        ):
        """builds agent network"""

        #observation
        inp = InputLayer((None,)+observation_shape,)
        #network body
        conv0 = Conv2DLayer(inp,32,3,stride=2,nonlinearity=elu)
        conv1 = Conv2DLayer(conv0,32,3,stride=2,nonlinearity=elu)
        conv2 = Conv2DLayer(conv1,32,3,stride=2,nonlinearity=elu)
        conv_flat = flatten(conv2)
        prev_gru = InputLayer((None,256))
        new_gru = GRUCell(prev_gru,conv_flat)

        dense1 = DenseLayer(concat([conv_flat,new_gru]),256,nonlinearity=tanh)
        policy_layer = DenseLayer(dense1,n_actions,nonlinearity=T.nnet.softmax)
        V_layer = DenseLayer(dense1,1,nonlinearity=None)


        #sample actions proportionally to policy_layer
        from agentnet.resolver import ProbabilisticResolver
        action_layer = ProbabilisticResolver(policy_layer)

        return Agent(observation_layers=inp,
                     policy_estimators=(policy_layer,V_layer),
                     agent_states={new_gru:prev_gru},
                     action_layers=action_layer)


    def make_train_fun(self,agent,
                       observation_shape=(1, 42, 42),  # same as env.observation_space.shape
                       sequence_length=20,  # how many steps to make before updating weights
                       reward_scale=10, #rewards are multiplied by this
                       gamma=0.99, #discount from TD
                       learning_rate=1e-4, #ADAM optimizer learning rate
        ):
        """Compiles a function to train for one step"""

        #make replay environment
        observations = T.tensor(theano.config.floatX,broadcastable=(False,)*(2+len(observation_shape)),
                                name="observations[b,t,color,width,height]")
        actions = T.imatrix("actions[b,t]")
        is_alive = T.imatrix("is_alive[b,t]")
        rewards = T.matrix("rewards[b,t]")
        prev_memory = [T.matrix("prev GRU[b,u]]")]


        replay = SessionBatchEnvironment(observations,
                                         [observation_shape],
                                         actions=actions,
                                         rewards=rewards,
                                         is_alive=is_alive)

        #replay sessions
        _, _, _, _, (policy_seq, V_seq) = agent.get_sessions(
            replay,
            session_length=sequence_length,
            experience_replay=True,
            initial_hidden=prev_memory
        )

        # get reference Qvalues according to Qlearning algorithm

        elwise_mse_loss = a2c_n_step.get_elementwise_objective(policy=policy_seq,
                                                               state_values=V_seq,
                                                               actions=replay.actions[0],
                                                               rewards=replay.rewards*reward_scale,
                                                               is_alive=replay.is_alive,
                                                               gamma_or_gammas=gamma,
                                                               n_steps=1,)

        reg = (1. / policy_seq).sum(axis=-1).mean()

        loss = elwise_mse_loss.mean() + 1e-4 * reg

        weights = get_all_params(agent.action_layers + list(agent.agent_states),trainable=True)

        # Compute weight updates
        updates = lasagne.updates.adam(loss, weights, learning_rate=learning_rate)

        # compile train function
        inputs = [observations, actions, rewards, is_alive]+prev_memory
        return theano.function(inputs, loss, updates=updates, allow_input_downcast=True)


    @lazy
    def train_fun(self):
        """compiles train_fun when asked."""
        print("Compiling train_fun on demand...")
        return self.make_train_fun(self.agent, sequence_length=self.sequence_length)

    def train_step(self,observations,actions,rewards,is_alive,prev_memory,*args,**kwargs):
        """Train on given batch (just call train_fun)"""
        return self.train_fun(observations,actions,rewards,is_alive,*prev_memory)




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
