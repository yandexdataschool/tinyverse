"""Few helper functions that you may find useful when building agents. Also used in examples"""


def lazy(fn):
    """
    Decorate to get lazy class fields (only computed when first demanded, if ever).
    used to avoid heavy compilation of training function when not training.

    ------example------
    >>>class myclass:
    >>>    @lazy
    >>>    def field(self):
    >>>        print("Obtaining field_value...")
    >>>        return 5
    >>>a = myclass()
    >>>print(a.field) #also prints "Obtaining field value"
    >>>print(a.field) #does not call field(self) again, uses stored value
    """
    attr_name = '_lazy_' + fn.__name__
    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop





import threading
import sys

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """

        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.

        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.

        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.

        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.

        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!

        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.


        ------example------

        >>>@background()
        >>>def iterate_minibatches(some_param):
        >>>    while True:
        >>>        X = read_heavy_file()
        >>>        X = do_helluva_math(X)
        >>>        y = wget_from_pornhub()
        >>>        do_pretty_much_anything()
        >>>        yield X_batch, y_batch

        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

#decorator
class background:
    def __init__(self,max_prefetch=1):
        self.max_prefetch = max_prefetch
    def __call__(self,gen):
        def bg_generator(*args,**kwargs):
            return BackgroundGenerator(gen(*args,**kwargs))
        return bg_generator

