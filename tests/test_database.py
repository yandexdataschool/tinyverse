from agentnet.experiments.openai_gym.pool import EnvPool
from lasagne.layers import InputLayer

import sys; sys.path.append("..")
from database import Database
from tqdm import tqdm
import gym
import numpy as np


def test_database():
    GAME = "Qbert-v0"
    env = gym.make(GAME)
    class mock_agent:
        #pseudo-agent that takes random actions
        agent_states={}
        observation_layers=[InputLayer((None,4,84,84))]
        action_layers=[InputLayer((None,))]

        @staticmethod    
        def step(obs):
            return ([env.action_space.sample() for _ in range(len(obs))],)

    pool = EnvPool(mock_agent,GAME,agent_step=mock_agent.step,preprocess_observation=lambda o: o[:84,:84,:])



    db = Database()
    for i in tqdm(xrange(1000)):
        observations,actions,rewards,memory,is_alive,info = pool.interact()
        db.record_session(observations[0],actions[0],rewards[0],is_alive[0],np.zeros(5))
