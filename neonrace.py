import gym
from gym import spaces
from universe.vectorized import ActionWrapper
from universe.wrappers import BlockingReset, Unvectorize, Vision
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode


from atari import PreprocessImage, AtariA3C
from tinyverse import Experiment


def make_experiment(db):
    """
    This is what's going to be created on "python tinyverse atari.py ..."
    """
    return AtariA3C(db)


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
                 client_id="vnc://localhost:5900+15900", #where to run universe VNC
                 keys = ('left', 'right', 'up', 'left up', 'right up', 'down', 'up x') #which keys can be pressed by agent
                 ):
        """a simple experiment setup that plays pong"""
        self.env_id = env_id
        self.client_id = client_id
        self.keys = keys
        
        agent = self.make_agent(observation_space=(1,64,64),n_actions=len(keys)) #we borrow agent structure from AtariA3C
        Experiment.__init__(self,db, agent, sequence_length=sequence_length)

    def make_env(self):
        """spawn a new environment instance"""

        env = gym.make(self.env_id)
        env = Vision(env)  # observation is an image
        env = BlockingReset(env)  # when env.reset will freeze until env is ready
        
        #convert from env.step(('KeyEvent', 'ArrowUp', True)) to env.step(2)
        env = DiscreteToFixedKeysVNCActions(env, self.keys) 
        env = Unvectorize(env) #now it's actually a single env instead of a batch

        # crop, grayscale and rescale to 64x64
        env = PreprocessImage(env,64,64,grayscale=True,
                              crop=lambda img: img[84:84 + 480, 18:18 + 640])
        
        env.configure(fps=5.0, remotes=1, start_timeout=15 * 60, client_id=self.client_id,
                      vnc_driver='go', vnc_kwargs={
                'encoding': 'tight', 'compress_level': 0,
                'fine_quality_level': 50, 'subsample_level': 3})

        return env


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




