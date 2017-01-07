"""A skeleton for player process"""

import numpy as np
from tqdm import tqdm
from itertools import count
from database import Database
db = Database()


#####Auxilary functions that may need their own module######
from lasagne.layers import get_all_param_values,set_all_param_values
def save_all_params(agent,name):
    all_params = get_all_param_values(list(agent.agent_states) + agent.action_layers)
    db.arctic['params'].write(name,all_params)

def load_all_params(agent,name):
    all_params = db.arctic['params'].read(name).data
    set_all_param_values(list(agent.agent_states) + agent.action_layers, all_params)


    
#####Main loop#####
from agentnet.experiments.openai_gym.pool import EnvPool
def generate_sessions(experiment,n_iters):
    agent = experiment.agent
    npz_file = np.load('action_layer.npz')
    set_all_param_values(agent.action_layers, npz_file['arr_0'])
    
    make_env = experiment.make_env
    seq_len = experiment.sequence_length
    
    pool = EnvPool(agent,lambda : make_env()) #TODO load new agentnet, rm lambda (bug is fixed)

    db = Database()
    if np.isinf(n_iters):
        epochs = count()
    else:
        epochs = range(n_iters)
    
    for i in tqdm(epochs):
        observations,actions,rewards,memory,is_alive,info = pool.interact(seq_len)
        db.record_session(observations[0],actions[0],rewards[0],is_alive[0],np.zeros(5))

    
    
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Player process. Example: python player.py "experiment" -n 1000')
    parser.add_argument('experiment', metavar='e', type=str,
                    help='a path to the experiment you wish to play')
    parser.add_argument('-n', dest='n_iters', type=int,default=float('inf'),
                    help='how many subsequences to record before exit (defaults to unlimited)')

    args = parser.parse_args()
    
    
    #load experiment from the specified module (TODO allow import by filepath)
    experiment = __import__(args.experiment).Experiment()
    
    
    #if there are any params, load them
    try: 
        load_all_params(experiment.agent,experiment.params_name)
    except:pass
    
    generate_sessions(experiment,args.n_iters)
    


