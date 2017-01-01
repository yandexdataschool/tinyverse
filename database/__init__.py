import time
import os
from arctic import Arctic
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

class Database:
    """
    mongoDB wrapper that supports basic operations with game sessions and params    
    """
    def __init__(self,
                 ip = "0.0.0.0",
                 port = 8900,
                 path = "./mongodb/",
                 sessions_db_name = "sessions",
                 params_db_name = "params",
                 default_quota = 100*(1024**3) #100gb
        ):
        
        hostname = "mongodb://{ip}:{port}".format(ip=ip,port=port)
        #see if db is even there
        try:
            Arctic(hostname,serverSelectionTimeoutMS=2000).list_libraries()
        except ServerSelectionTimeoutError:
            
            #if not, launch
            starter = "mkdir -p {path} && nohup mongod --bind_ip {ip} --port {port} --dbpath {path} &".format(
                    ip=ip,port=port,path=path
                )
            
            #how long to wait: 30s first time, 10s afterwards
            dt = 30 if not os.path.exists(path) else 10
                
            print "db not found at {}, launching({}s)...".format(hostname,dt)
            os.system(starter) #intentionally starting non-daemon process
            time.sleep(dt)

        
        #init arctic
        self.arctic = Arctic(hostname,app_name="tensors")        
        
        #get or initialize sessions & params databases
        if params_db_name not in self.arctic.list_libraries():
            print "creating", params_db_name
            self.arctic.initialize_library(params_db_name)
            self.arctic.set_quota(params_db_name,default_quota)
            
        self.params = self.arctic[params_db_name]

        if sessions_db_name not in self.arctic.list_libraries():
            print "creating", sessions_db_name
            self.arctic.initialize_library(sessions_db_name)
            self.arctic.set_quota(sessions_db_name,default_quota)
        
        self.sessions = self.arctic[sessions_db_name]
        
        #init regular databases
        self.registry = MongoClient(hostname)["main"][sessions_db_name]
                
    
    #########################
    ###public DB interface###
    #########################
    #however you implement me, i must have these methods:
    
    def record_session(self,observations,actions,rewards,is_alive,initial_memory,prev_session_index=-1):
        """
        Creates database entry for a single game session.
        Game session needn't start from beginning or end at the terminal state.
        """
        index = self.registry.insert_one({"prev_session_index":prev_session_index}).inserted_id
        
        keys = ["observations","actions","rewards","is_alive","initial_memory"]
        values = [observations,actions,rewards,is_alive,initial_memory]
        
        for key,value in zip(keys,values):
            self.sessions.write("{}.{}".format(index,key),value)

        return index
    
    def get_session(self,index):
        """
        obtains all the data for a particular session
        :returns: observations,actions,rewards,is_alive,initial_memory
        """
        keys = ["observations","actions","rewards","is_alive","initial_memory"]
        return [self.sessions.read("{}.{}".format(index,key)).data for key in keys]
    
    ##something about storing and loading parms##
