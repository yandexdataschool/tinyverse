
import redis
import os,sys,time
from warnings import  warn
from six.moves import cPickle as pickle
from lasagne.layers import get_all_param_values, set_all_param_values

class Database:
    def __init__(self,
                 host="localhost",
                 port=7070,
                 default_prefix="",
                 default_session_key="sessions",
                 default_params_key="weights",
                 default_worker_prefix="workers.",
                 *args,**kwargs
                 ):
        """
        The database instance that can stores sessions,weights and metadata.
        Implemented as a thin wrapper for Redis.

        :param host: redis hostname
        :param port: redis port
        :param args: args for database client (redis)
        :param kwargs: kwargs for database client (redis), e.g. password="***"
        :param default_prefix: prepended to both default_params_key and default_session_key and 
            default_worker_prefix. Does NOT affect custom keys.
        :param default_session_key: default name for session list
        :param default_params_key: default name for weights pickle
        """

        # if localhost and can't find redis, start one
        if host in ("localhost","0.0.0.0", "127.0.0.1", "*"):
            try:
                redis.Redis(host=host, port=port, *args,**kwargs).client_list()
            except redis.ConnectionError:
                # if not, on localhost try launch new one
                print("Redis not found on %s:%s. Launching new redis..." % (host, port))
                self.start_redis(port)
                time.sleep(5)

        self.redis = redis.Redis(host=host, port=port, *args,**kwargs)

        #naming parameters
        self.default_session_key = default_prefix+default_session_key
        self.default_params_key = default_prefix+default_params_key
        self.default_worker_prefix= default_prefix+default_worker_prefix


    def start_redis(self, port=7070):
        """starts a redis serven in a NON-DAEMON mode"""
        os.system("nohup redis-server --port %s > .redis.log &" % port)
        
    def worker_heartbeat(self,role,pid=None,worker_prefix=None,value=1,expiration_time=30):
        """registers worker to the database. Used to quickly kill all workers."""
        pid= pid or os.getpid()
        worker_prefix = worker_prefix or self.default_worker_prefix
        key = worker_prefix+role+"."+str(pid)
        self.redis.set(key,value)
        if expiration_time is not None:
            self.redis.expire(key,expiration_time)

    def dumps(self,data):
        """converts whatever to string"""
        return pickle.dumps(data, protocol=2)

    def loads(self,string):
        """converts string to whatever was dumps'ed in it"""
        kwargs = {}
        if sys.version_info >= (3,):
            kwargs['encoding'] = 'latin1'
        return pickle.loads(string, **kwargs)

    #########################
    ###public DB interface###
    #########################
    # however you implement me, i must have these methods:

    def record_session(self, observations, actions, rewards, is_alive,
                       initial_memory=None,
                       session_key=None,
                       index=None):
        """
        Creates database entry for a single game session.
        Game session needn't start from beginning or end at the terminal state.
        """
        self.worker_heartbeat("play")

        session_key = session_key or self.default_session_key

        data = self.dumps([observations, actions, rewards, is_alive, initial_memory])

        if index is None:
            self.redis.lpush(session_key, data)
        else:
            self.redis.lset(session_key, index, data)
            

    def num_sessions(self, session_key=None):
        """returns number of sessions under specified prefix"""
        session_key = session_key or self.default_session_key
        return self.redis.llen(session_key)

    def get_session(self, index=0, session_key=None, ):
        """
        obtains all the data for a particular session
        :returns: observations,actions,rewards,is_alive,initial_memory
        """
        self.worker_heartbeat("train")
        
        session_key = session_key or self.default_session_key
        data = self.redis.lindex(session_key, index)
        return self.loads(data)

    def trim_sessions(self, start=0, end=-1, session_key=None):
        """
        removes the data for all sessions but for given range
        Both start and end are INCLUSIVE borders
        """
        session_key = session_key or self.default_session_key
        self.redis.ltrim(session_key, start, end)

    def save_all_params(self, agent, key=None):
        """saves agent params into the database under given name. overwrites by default"""
        key = key or self.default_params_key
        all_params = get_all_param_values(list(agent.agent_states) + agent.action_layers)
        self.redis.set(key, self.dumps(all_params))

    def load_all_params(self, agent, key=None,errors='raise'):
        """loads agent params from the database under the given name"""
        assert errors in ('raise', 'warn', 'ignore'), "errors must be 'raise','warn' or 'ignore'"

        if errors == 'raise':
            #Main function
            key = key or self.default_params_key
            raw = self.redis.get(key)
            if raw is None:
                raise redis.ResponseError("Params not found under key '%s' (got None)" % key)

            all_params = self.loads(raw)
            set_all_param_values(list(agent.agent_states) + agent.action_layers, all_params)

        else:
            #Error handling
            try:
                return self.load_all_params(agent,key=key,errors='raise')
            except:
                exc_type, exc, tb = sys.exc_info()
                if errors == 'warn':
                    warn(str(exc))




