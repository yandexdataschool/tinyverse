import gym
from gym import spaces
from universe.vectorized import ActionWrapper
from universe.wrappers import BlockingReset, Unvectorize, Vision
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode


from atari import PreprocessImage, AtariA3C
from tinyverse import Experiment,lazy
import theano
import theano.tensor as T
import lasagne
from agentnet.learning import a2c
from agentnet.environment import SessionBatchEnvironment


def make_experiment(db):
    """
    This is what's going to be created on "python tinyverse atari.py ..."
    """
    return UniverseA3C(db)


class UniverseA3C(AtariA3C):
    """
    A class that defines the reinforcement learning experiment.

    Since we approximately the same image size and action space,
    we inherit network and learning algo from pong agent (atari.py)


    It can than be sent playing/training/evaluating via
    - python ./tinyverse neonrace.py play
    - python ./tinyverse neonrace.py train -b 10
    - python ./tinyverse neonrace.py eval -n 5


    """

    def __init__(self,
                 db,  # database instance (mandatory parameter)
                 sequence_length=25,  # how many steps to make before updating weights
                 env_id='flashgames.NeonRace-v0',  # which game to play (uses gym.make)
                 client_id=None,#"vnc://localhost:5900+15900", #where to run universe VNC
                 keys = ('left', 'right', 'up', 'left up', 'right up', 'down', 'up x') #which keys can be pressed by agent
                 ):
        """a simple experiment setup that plays pong"""
        self.env_id = env_id
        self.client_id = client_id
        self.keys = keys
        
        agent = self.make_agent(observation_shape=(1,64,64),n_actions=len(keys)+1) #we borrow agent structure from AtariA3C
        Experiment.__init__(self,db, agent, sequence_length=sequence_length)

    def make_env(self):
        """spawn a new environment instance"""
        print(self.env_id)
        env = gym.make(self.env_id)
        env = Vision(env)  # observation is an image
        env = BlockingReset(env)  # when env.reset will freeze until env is ready
        
        #convert from env.step(('KeyEvent', 'ArrowUp', True)) to env.step(2)
        env = DiscreteToFixedKeysVNCActions(env, list(self.keys) )
        env = Unvectorize(env) #now it's actually a single env instead of a batch

        # crop, grayscale and rescale to 64x64
        env = PreprocessImage(env,64,64,grayscale=True,
                              crop=lambda img: img[84:84 + 480, 18:18 + 640])
        
        env.configure(fps=5.0, remotes=1, start_timeout=15 * 60, client_id=self.client_id,
                      vnc_driver='go', vnc_kwargs={
                'encoding': 'tight', 'compress_level': 0,
                'fine_quality_level': 50, 'subsample_level': 3})

        return env
    
    def make_train_fun(self,agent,
                       sequence_length=25,  # how many steps to make before updating weights
                       observation_shape=(1,64, 64),  # same as env.observation_space.shape
                       reward_scale=1e-3, #rewards are multiplied by this. May be useful if they are large.
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
        loss = 0.1*elwise_actor_loss.mean() + 0.25*elwise_critic_loss.mean() + 1e-3*reg_entropy + 1e-2*reg_logits

        
        # Compute weight updates, clip by norm
        grads = T.grad(loss,self.weights)
        grads = lasagne.updates.total_norm_constraint(grads,10)
        
        updates = lasagne.updates.adam(grads, self.weights,1e-4)


        # compile train function
        inputs = [observations, actions, rewards, is_alive]+prev_memory
        return theano.function(inputs,
                               updates=rng_updates+updates,
                               allow_input_downcast=True)


class FixedKeyState(object):
    def __init__(self, keys):
        self._keys = [keycode(key) for key in keys]
        self._down_keysyms = set()

    def apply_vnc_actions(self, vnc_actions):
        for event in vnc_actions:
            if isinstance(event, vnc_spaces.KeyEvent):
                if event.down:
                    self._down_keysyms.add(event.key)
                else:
                    self._down_keysyms.discard(event.key)

    def to_index(self):
        action_n = 0
        for key in self._down_keysyms:
            if key in self._keys:
                # If multiple keys are pressed, just use the first one
                action_n = self._keys.index(key) + 1
                break
        return action_n


class DiscreteToFixedKeysVNCActions(ActionWrapper):
    """
    Define a fixed action space. Action 0 is all keys up. Each element of keys can be a single key or a space-separated list of keys

    For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right'])
    will have 3 actions: [none, left, right]

    You can define a state with more than one key down by separating with spaces. For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right', 'space', 'left space', 'right space'])
    will have 6 actions: [none, left, right, space, left space, right space]
    """

    def __init__(self, env, keys):
        super(DiscreteToFixedKeysVNCActions, self).__init__(env)

        self._keys = keys
        self._generate_actions()
        self.action_space = spaces.Discrete(len(self._actions))

    def _generate_actions(self):
        self._actions = []
        uniq_keys = set()
        for key in self._keys:
            for cur_key in key.split(' '):
                uniq_keys.add(cur_key)

        for key in [''] + self._keys:
            split_keys = key.split(' ')
            cur_action = []
            for cur_key in uniq_keys:
                cur_action.append(vnc_spaces.KeyEvent.by_name(cur_key, down=(cur_key in split_keys)))
            self._actions.append(cur_action)
        self.key_state = FixedKeyState(uniq_keys)

    def _action(self, action_n):
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        return [self._actions[int(action)] for action in action_n]




