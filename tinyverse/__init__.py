"""Import-throughs and miscelaneous stuff."""

__version__=1.0

from database import Database
from experiment import Experiment


#helpers
def lazy(fn):
    """
    decorates property to make it run only when first demanded (if ever).
    used to avoid heavy compilation of training function when not training.
    """
    attr_name = '_lazy_' + fn.__name__
    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


