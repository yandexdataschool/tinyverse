"""a minimalistic experiment for atari skiing"""

import gym
import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.regularization import regularize_network_params,l2

from agentnet import Agent
from agentnet.environment import SessionBatchEnvironment
from agentnet.learning import a2c

from tinyverse import Experiment, lazy
from prefetch_generator import background

def make_experiment(db):
    """
    This is what's going to be created on "python tinyverse atari.py ..."
    """
    return AtariA3C(db)


class AtariA3C(Experiment):
    """
    A class that defines the reinforcement learning experiment.
    
    This particular experiment implements a simple convolutional network with A3C algorithm.
    
    It can than be sent playing/training/evaluating via
    - python ./tinyverse atari.py play
    - python ./tinyverse atari.py train -b 10
    - python ./tinyverse atari.py eval -n 5
    
    
    """
    def __init__(self,
                 db, #database instance (mandatory parameter)
                 sequence_length=25,  # how many steps to make before updating weights
                 ):
        """a simple experiment setup that plays pong"""
        super(AtariA3C, self).__init__(db, self.make_agent(), sequence_length=sequence_length)

    def make_env(self):
        """spawn a new environment instance"""
        env = gym.make("Skiing-v0")
        env = RewardForPassingGates(env)
        env = PreprocessImage(env,crop=lambda img:img[50:-20,10:-10],grayscale=True)
        return env
    
    def make_agent(self,
                   observation_shape=(1, 64, 64), # same as env.observation_space.shape
                   n_actions = 3,  # same as env.action_space.n
        ):
        """builds agent network"""

        #observation
        inp = InputLayer((None,)+observation_shape,)

        #4-tick window over images
        from agentnet.memory import WindowAugmentation
        prev_wnd = InputLayer((None,4)+observation_shape)
        new_wnd = WindowAugmentation(inp,prev_wnd)
        
        #reshape to (channels, h,w). If you don't use grayscale, 4 should become 12.
        wnd_reshape = reshape(new_wnd, (-1,4)+observation_shape[1:])

        #network body
        conv0 = Conv2DLayer(wnd_reshape,32,5,stride=2,nonlinearity=elu)
        conv1 = Conv2DLayer(conv0,32,5,stride=2,nonlinearity=elu)
        conv2 = Conv2DLayer(conv1,64,5,stride=1,nonlinearity=elu)
        
        dense = DenseLayer(dropout(conv2,0.1),512,nonlinearity=tanh)
        
        #actor head
        logits_layer = DenseLayer(dense,n_actions,nonlinearity=None) 
        #^^^ store policy logits to regularize them later
        policy_layer = NonlinearityLayer(logits_layer,T.nnet.softmax)
        
        #critic head
        V_layer = DenseLayer(dense,1,nonlinearity=None)
        
        #sample actions proportionally to policy_layer
        from agentnet.resolver import ProbabilisticResolver
        action_layer = ProbabilisticResolver(policy_layer)
        
        #get all weights (just like any lasagne network). new_out mentioned just in case.
        self.weights = get_all_params([V_layer,policy_layer],trainable=True)


        return Agent(observation_layers=inp,
                     policy_estimators=(logits_layer,V_layer),
                     agent_states={new_wnd:prev_wnd},
                     action_layers=action_layer)


    def make_train_fun(self,agent,
                       sequence_length=25,  # how many steps to make before updating weights
                       observation_shape=(1,64, 64),  # same as env.observation_space.shape
                       reward_scale=1, #rewards are multiplied by this. May be useful if they are large.
                       gamma=0.99, #discount from TD
        ):
        """Compiles a function to train for one step"""

        #make replay environment
        observations = T.tensor(theano.config.floatX,broadcastable=(False,)*(2+len(observation_shape)),
                                name="observations[b,t,color,width,height]")
        
        actions = T.imatrix("actions[b,t]")
        rewards,is_alive = T.matrices("rewards[b,t]","is_alive[b,t]")
        prev_memory = [l.input_var for l in agent.agent_states.values()]


        replay = SessionBatchEnvironment(observations,
                                         [observation_shape],
                                         actions=actions,
                                         rewards=rewards,
                                         is_alive=is_alive)

        #replay sessions
        _, _, _, _, (logits_seq, V_seq) = agent.get_sessions(
            replay,
            session_length=sequence_length,
            experience_replay=True,
            initial_hidden=prev_memory,
            unroll_scan=False,#speeds up compilation 10x, slows down training by 20% (still 4x faster than TF :P )
        )
        rng_updates = agent.get_automatic_updates() #updates of random states (will be passed to a function)
        
        # compute pi(a|s) and log(pi(a|s)) manually [use logsoftmax]
        # we can't guarantee that theano optimizes logsoftmax automatically since it's still in dev
        logits_flat = logits_seq.reshape([-1,logits_seq.shape[-1]])
        policy_seq = T.nnet.softmax(logits_flat).reshape(logits_seq.shape)
        logpolicy_seq = T.nnet.logsoftmax(logits_flat).reshape(logits_seq.shape)
        
        # get policy gradient
        elwise_actor_loss,elwise_critic_loss = a2c.get_elementwise_objective(policy=logpolicy_seq,
                                                                             treat_policy_as_logpolicy=True,
                                                                             state_values=V_seq[:,:,0],
                                                                             actions=replay.actions[0],
                                                                             rewards=replay.rewards*reward_scale,
                                                                             is_alive=replay.is_alive,
                                                                             gamma_or_gammas=gamma,
                                                                             n_steps=None,
                                                                             return_separate=True)
        
        # add losses with magic numbers 
        # (you can change them more or less harmlessly, this usually just makes learning faster/slower)
        # also regularize to prioritize exploration
        reg_logits = T.mean(logits_seq**2)
        reg_entropy = T.mean(T.sum(policy_seq*logpolicy_seq,axis=-1))
        loss = 0.1*elwise_actor_loss.mean() + 0.25*elwise_critic_loss.mean() + 5e-4*reg_entropy + 1e-4*reg_logits

        
        # Compute weight updates, clip by norm
        grads = T.grad(loss,self.weights)
        grads = lasagne.updates.total_norm_constraint(grads,5)
        
        updates = lasagne.updates.adam(grads, self.weights,1e-4)


        # compile train function
        inputs = [observations, actions, rewards, is_alive]+prev_memory
        return theano.function(inputs,
                               updates=rng_updates+updates,
                               allow_input_downcast=True)

    
    def train_step(self,observations,actions,rewards,is_alive,prev_memory,*args,**kwargs):
        """Train on given batch (just call train_fun)"""
        
        self.train_fun(observations,actions,rewards,is_alive,*prev_memory)
        
    #some optimizations
    
    @lazy
    def train_fun(self):
        """compiles train_fun when asked. Used to NOT waste time on that in the player process (~10-15s at the start)"""
        print("Compiling train_fun on demand...")
        train_fun = self.make_train_fun(self.agent, sequence_length=self.sequence_length)
        print("Done!")
        return train_fun
    
    @background(max_prefetch=10)
    def iterate_minibatches(self,*args,**kwargs):
        """makes minibatch iterator work in a separate thread (speedup ~20%). Also prints RPS via tqdm."""
        from tqdm import tqdm
        return tqdm(super(AtariA3C,self).iterate_minibatches(*args,**kwargs))




import cv2
from gym.spaces.box import Box
from gym.core import ObservationWrapper
class PreprocessImage(ObservationWrapper):
    def __init__(self,env,height=64,width=64,grayscale=True,
                 crop=lambda img: img[34:34+160]):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height,width)
        self.grayscale = grayscale
        self.crop=crop
        
        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors,height,width])
        
    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = cv2.resize(img, self.img_size)
        if self.grayscale:
            img=img.mean(-1,keepdims=True)
        img = np.transpose(img,(2,0,1)) #reshape from (h,w,colors) to (colors,h,w)
        img = img.astype('float32')/255.
        return img
    
from gym.core import Wrapper
class RewardForPassingGates(Wrapper):
    def _reset(self):
        """On game reset, remember the hash of initial score"""
        s = self.env.reset()
        self.prev_score_hash = hash(s[31:38,67:81].tobytes()) #hash of the image chunk with scoreboard
        return s
    def _step(self,action):
        """on each step, if score has changed, give +1 reward, else +0"""
        s,_,done,info = self.env.step(action)
        new_score_hash = hash(s[31:38,67:81].tobytes()) #hash of the same image chunk
        
        #reward = +1 if we have just crossed the gate, else 0
        r = int(new_score_hash != self.prev_score_hash)
        
        #remember new score
        self.prev_score_hash = new_score_hash
        return s,r,done,info
