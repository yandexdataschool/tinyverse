"""A skeleton for player process"""

import numpy as np
from tqdm import tqdm
from itertools import count
from database import Database
db = Database()


#####Main loop#####
from agentnet.experiments.openai_gym.pool import EnvPool
def evaluate(experiment, n_games):
    agent = experiment.agent
    make_env = experiment.make_env
    pool = EnvPool(agent,make_env)

    r = pool.evaluate(n_games)
    
    
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Player process. Example: python player.py "experiment" -n 1000')
    parser.add_argument('experiment', metavar='e', type=str,
                    help='a path to the experiment you wish to play')
    parser.add_argument('-n', dest='n_games', type=int,default=1,
                    help='how many games to evaluate on')

    args = parser.parse_args()
    
    
    #load experiment from the specified module (TODO allow import by filepath)
    experiment = __import__(args.experiment).Experiment()
    
    evaluate(experiment, args.n_games)
    


