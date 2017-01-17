"""a minimalistic experiment designed to test the framework"""
import time
from itertools import count
import numpy as np
from agentnet.experiments.openai_gym.pool import EnvPool

class Experiment(object):
    def __init__(self,
                 db,
                 agent,
                 sequence_length,
        ):
        """
        The base experiment that does all the generic stuff (playing, training).
        :param db: database (mandatory first param)
        :param agent: agentnet.agent.Agent instance to train, save/load, etc.
        :param sequence_length: how many iterations to make per 1 weight update.
        """

        self.agent = agent
        self.sequence_length = sequence_length
        self.db=db


    def make_env(self):
        """spawn a new environment instance"""
        raise NotImplementedError("please implement make_env")

    def train_step(self, observations,actions,rewards,is_alive,prev_memory,*args,**kwargs):
        """
        Train for one step on a small batch of data
        :param observations: observations as received from environment [batch,tick,*obs shape]
        :param actions: agent actions as provided by agent, [batch,tick,*action shape]
        :param rewards: rewards matrix for each [batch,tick]
        :param is_alive: a matrix[batch,tick] indicating whether agent is alive at current tick. 0 means session finished.
        :param prev_memory: a list of agent memory tensors (window, lstm, gru, whatever) as they were before tick 0
        """
        raise NotImplementedError("please implement train_step")

    def generate_sessions(self, n_iters=float('inf'), n_games=1, reload_period=10):
        """
        Generates sessions and records them to the database
        :param n_iters: how many batches to generate (inf means generate forever)
        :param n_games: how many games to maintain in parallel
        :param reload_period: how often to read weights from database (every reload_period batches)
        """

        pool = EnvPool(self.agent, self.make_env, n_games=n_games)

        loop = count() if np.isinf(n_iters) else range(n_iters)
        
        try:
            self.db.load_all_params(self.agent)
        except:
            self.db.save_all_params(self.agent)

        for epoch in loop:
            if (epoch+1) % reload_period == 0:
                self.db.load_all_params(self.agent)

            # play
            prev_memory = list(pool.prev_memory_states)
            observations, actions, rewards, memory, is_alive, info = pool.interact(self.sequence_length)

            # save sessions
            for k in range(n_games):
                self.db.record_session(observations[k], actions[k], rewards[k], is_alive[k],
                                       [mem[k] for mem in prev_memory])


    def iterate_minibatches(self, n_iters=float('inf'), batch_size=100, replay_buffer_size=5000, trim_every=100):
        """
        Sample batch_size random sessions from database for n_iters
        :param n_iters: how many iterations (minibatches) to train for
        :param batch_size: how many examples to take per minibatch
        :param replay_buffer_size: how many [newest] batches to maintain (trim all older than that)
        :param trim_every: how often to trim old experience (every trim_every batches)
        """
        epochs = count() if np.isinf(n_iters) else range(n_iters)
        for epoch in epochs:

            # sample random indices
            batch_keys = np.random.randint(0, self.db.num_sessions(), batch_size)

            # sample batch by these indices
            batch = list(map(self.db.get_session, batch_keys))
            
            #meld everything to numpy arrays and yield it
            obs,act,rw,alive,mem = zip(*batch)
            obs,act,rw,alive = map(np.stack,[obs,act,rw,alive])
            mem = list(map(np.stack,zip(*mem)))
            yield obs,act,rw,alive,mem

            # trim pool
            if epoch % trim_every == 0:
                self.db.trim_sessions(0, replay_buffer_size)


    def train_on_sessions(self, n_iters=float('inf'), batch_size=100,
                          replay_buffer_size=5000,save_period=10,
                          wait_for_sessions=True):
        """
        Train on sessions from database for n_iters
        :param n_iters: how many iterations (minibatches) to train for
        :param batch_size: how many examples to take per minibatch
        :param replay_buffer_size: how many [newest] batches to maintain (trim all older than that)
        :param save_period: how often to save weights to the database (every save_period minibatches)
        :param wait_for_sessions: if True, sleeps until there are nonzero game sessions
        """

        if wait_for_sessions:
            while self.db.num_sessions() == 0:
                print("Awaiting sessions...")
                time.sleep(5)


        # load params
        self.db.load_all_params(self.agent,errors='warn')

        iterator = self.iterate_minibatches(n_iters, batch_size, replay_buffer_size)

        for epoch, batch in enumerate(iterator):
            if (epoch+1) % save_period == 0:
                self.db.save_all_params(self.agent)
                
            self.train_step(*batch) # feed a batch of (obs, act, rew, is_alive, prev_mem)


    def evaluate(self, n_games,*args,**kwargs):
        """
        Play several full games and averages agent rewards. Prints some info unless verbose=False
        :param n_games: how many games to play (successively without changing weights)
        """
        self.db.load_all_params(self.agent, errors='warn')
        return EnvPool(self.agent,self.make_env,
                       n_games=0).evaluate(n_games,*args,**kwargs)



