"""A skeleton for player process"""

import numpy as np
from tqdm import tqdm
from itertools import count
from database import Database
db = Database()

import theano
import theano.tensor as T
from agentnet.environment import SessionBatchEnvironment

def sample_batch(batch_size):
    """sample batch_size random sessions from database"""
    batch_keys = np.random.randint(0,db.num_sessions(),batch_size)
        
    batch = []
    for key in batch_keys:
        batch.append(db.get_session(key))
        
    return zip(*batch)




from prefetch_generator import background
@background(max_prefetch=10)
def iterate_minibatches(n_iters,batch_size,replay_buffer_size,trim_every=100):
    epochs = count() if np.isinf(n_iters) else range(n_iters)
    for epoch in epochs:
        yield sample_batch(batch_size)
        if epoch % trim_every ==0:
            db.trim_sessions(0,replay_buffer_size)

    
    
    
def train_on_sessions(experiment,batch_size,n_iters,
                      save_period=100,replay_memory_size=50000):
    
    
    observations = T.tensor5("observations[b,t,u]")
    actions = T.imatrix("actions[b,t]")
    is_alive = T.imatrix("is_alive[b,t]")
    rewards = T.matrix("rewards[b,t]")
    #TODO add prev_memory_states
    
    inputs = [observations,actions,rewards,is_alive]
    
    replay_env = SessionBatchEnvironment(observations,[(1,42,42)], ###FIX hard-coded
                                         actions=actions,
                                         rewards=rewards,
                                         is_alive=is_alive)
    
    train_step = experiment.build_train_step(replay_env,inputs)
    
    #load params
    db.load_all_params(experiment.agent,experiment.params_name,errors='warn')
    
    iterator = iterate_minibatches(n_iters,batch_size,replay_memory_size)
    
    for i,batch in tqdm(enumerate(iterator)):
        if i % save_period == 0 or (i == 0 and np.isinf(reload_period)):
            db.save_all_params(experiment.agent, experiment.params_name,errors='warn')
            
        s,a,r,alive,_ = batch
        train_step(s,a,r,alive)
            


    


    
    
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Player process. Example: python learner.py experiment -n 1000')
    parser.add_argument('experiment', metavar='e', type=str,
                    help='a path to the experiment you wish to play')
    parser.add_argument('-n', dest='n_iters', type=int,default=float('inf'),
                    help='how many subsequences to record before exit (defaults to unlimited)')
    parser.add_argument('-b', dest='batch_size', type=int,default=1,
                    help='how many sessions to sample on each training iteration')
    parser.add_argument('-s', dest='save_period', type=int,default=100,
                    help='period (in epochs), how often NN weights are going to be saved')

    args = parser.parse_args()
    
    
    #load experiment from the specified module (TODO allow import by filepath)
    experiment = __import__(args.experiment).Experiment()
    
    train_on_sessions(experiment, args.batch_size, args.n_iters, args.save_period)
    


