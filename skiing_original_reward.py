"""just like atari.py, but no reward tricks. Used for evaluation"""

def make_experiment(db):
    """
    This is what's going to be created on "python tinyverse atari.py ..."
    """
    return PureAtariA3C(db)

import gym
from atari import AtariA3C, PreprocessImage
class PureAtariA3C(AtariA3C):
    def make_env(self):
        """Spawn a new environment instance.
        This class does NOT use custom reward wrapper.
        Used for evaluation"""
        env = gym.make("Skiing-v0")
        env = PreprocessImage(env,crop=lambda img:img[50:-20,10:-10],grayscale=True)
        return env
