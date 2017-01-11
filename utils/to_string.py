
import sys
from six.moves import cPickle as pickle
def dumps(data):
    """converts whatever to string"""
    return pickle.dumps(data,protocol=2)

def loads(string):
    """converts string to whatever was dumps'ed in it"""
    kwargs={}
    if sys.version_info >= (3,):
        kwargs['encoding']= 'latin1'
        return pickle.loads(string,**kwargs)
