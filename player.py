"""A skeleton for player process"""

import numpy as np
from tqdm import tqdm
from itertools import count
from database import Database
db = Database()


#####Main loop#####
from agentnet.experiments.openai_gym.pool import EnvPool
def generate_sessions(experiment, n_iters, n_games,reload_period):
    agent = experiment.agent
    
    make_env = experiment.make_env
    seq_len = experiment.sequence_length
    
    pool = EnvPool(agent,make_env,n_games=n_games)

    if np.isinf(n_iters):
        epochs = count()
    else:
        epochs = range(n_iters)
    
    for i in tqdm(epochs):
        if i % reload_period == 0 or (i == 0 and np.isinf(reload_period)):
            db.load_all_params(agent, experiment.params_name,errors='warn')
            
            
        prev_memory = list(pool.prev_memory_states)
        observations,actions,rewards,memory,is_alive,info = pool.interact(seq_len)
        
        for k in range(n_games):
            db.record_session(observations[k],actions[k],rewards[k],is_alive[k],
                              [mem[k] for mem in prev_memory])

    
    
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Player process. Example: python player.py "experiment" -n 1000')
    parser.add_argument('experiment', metavar='e', type=str,
                    help='a path to the experiment you wish to play')
    parser.add_argument('-n', dest='n_iters', type=int,default=float('inf'),
                    help='how many subsequences to record before exit (defaults to unlimited)')
    parser.add_argument('-g', dest='n_games', type=int,default=1,
                    help='how many games to play in parallel')
    parser.add_argument('-r', dest='reload_period', type=int,default=100,
                    help='period (in epochs), how often NN weights are going to be reloaded')

    args = parser.parse_args()
    
    
    #load experiment from the specified module (TODO allow import by filepath)
    experiment = __import__(args.experiment).Experiment()
    
    generate_sessions(experiment, args.n_iters, args.n_games, args.reload_period)
    


