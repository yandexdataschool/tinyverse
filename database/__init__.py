import os
import redis

from lasagne.layers import get_all_param_values,set_all_param_values

import numpy as np
from utils import error_handling
from utils.to_string import loads,dumps



class Database:
    """
    mongoDB wrapper that supports basic operations with game sessions and params    
    """
    def __init__(self,
                 host = "0.0.0.0",
                 port = 7070,
                 password=None,
                 default_session_key="sessions.",
        ):
        
        
        self.redis = redis.Redis(host=host,port=port,password=password)
        self.default_session_key=default_session_key
        
        
    #########################
    ###public DB interface###
    #########################
    #however you implement me, i must have these methods:
    
    def record_session(self,observations,actions,rewards,is_alive,
                       initial_memory=None,
                       prev_session_index=-1,
                       session_key=None,
                       index=None):
        """
        Creates database entry for a single game session.
        Game session needn't start from beginning or end at the terminal state.
        """
        
        session_key = session_key or self.default_session_key
        
        data = dumps([observations,actions,rewards,is_alive,initial_memory])

        if index is None:
            self.redis.lpush(session_key,data)
        else:
            self.redis.lset(session_key,index,data)
        
    def num_sessions(self,session_key=None):
        """returns number of sessions under specified prefix"""
        session_key = session_key or self.default_session_key
        return self.redis.llen(session_key)
        
    def get_session(self,index=0,session_key=None,):
        """
        obtains all the data for a particular session
        :returns: observations,actions,rewards,is_alive,initial_memory
        """
        session_key = session_key or self.default_session_key
        data = self.redis.lindex(session_key,index)
        return loads(data)

    def trim_sessions(self,start=0,end=-1,session_key=None):
        """
        removes the data for all sessions but for given range
        Both start and end are INCLUSIVE borders
        """
        session_key = session_key or self.default_session_key
        self.redis.ltrim(session_key,start,end)
            
        
        
        
        
    
    @error_handling
    def save_all_params(self,agent,name):
        """saves agent params into the database under given name. overwrites by default"""
        all_params = get_all_param_values(list(agent.agent_states) + agent.action_layers)
        self.redis.set(name,dumps(all_params))
                

    @error_handling
    def load_all_params(self,agent,name):
        """loads agent params from the database under the given name"""
        all_params = loads(self.redis.get(name))
        set_all_param_values(list(agent.agent_states) + agent.action_layers, all_params)

