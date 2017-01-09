"""non-specific helper functions"""

import sys
from warnings import warn
def error_handling(f):
    """allows user to select error handling mode:
     - errors='raise' - just calls the function
     - errors='warn' - converts errors to warnings
     - errors='ignore' - exits function silently on error
    """
    def f_safe(*args,**kwargs):
        errors = kwargs.pop('errors','raise')
        
        if errors == 'raise':
            return f(*args,**kwargs)
        else:
            try:
                return f(*args,**kwargs)
            except:
                
                exc_type,exc,tb = sys.exc_info()
                if errors =='warn':
                    warn(exc.message)
                else:
                    assert errors =='ignore', "errors must be 'raise','warn' or 'ignore'"
    f_safe.__doc__=f.__doc__
    return f_safe
