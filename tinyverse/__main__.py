"""
Runner script for tinyverse.
"""
import sys
sys.path.insert(0,".")

import argparse
import imp
from traceback import print_exception
import numpy as np
from database import Database

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tinyverse main script.'
                                                 'Run with python tinyverse --whatever')
    parser.add_argument('path', metavar='path', type=str,
                        help='a path to the experiment you wish to use. (e.g. breakout.py)')

    parser.add_argument('mode', metavar='mode', type=str,
                        help='what to do with an experiment:\n'
                             '- "play" - add new sessions (-n iters -b parallel games -s iters between sync)\n'
                             '- "train" - train the agent on recorded sessions (-n iters -b batch size -s iters between sync)\n'
                             '- "eval" - evaluate the agent on several full games (-n full games)\n'
                             '- "info" - display dashboard\n'
                             '- "clear" - remove all recorded sessions and weights\n')

    #mode parameters
    parser.add_argument('-n', dest='n_iters', type=int, default=float('inf'),
                        help='how many minibatches to record/process before exit, used in "play", "train", "eval".'
                             'In "eval" mode it means the amount of games to play at all')
    parser.add_argument('-b', dest='batch_size', type=int, default=1,
                        help='how many sessions to play/learn in parallel. Used in "play" and "train"')

    parser.add_argument('-s', dest='sync_period', type=int, default=10,
                        help='period (in minibatches), how often NN weights should be loaded in "play" or saved in "train".')

    parser.add_argument('--buffer-size', dest='buffer-size', type=int, default=5000,
                        help='The amount of sessions stored in the database (if exceeded, oldest records are trimmed in "train")')

    #database parameters
    parser.add_argument('--port',dest='port',type=int,default=7070,
                        help = 'database port')
    parser.add_argument('--host',dest='host',type=str,default="localhost",
                        help = 'database host')
    parser.add_argument('--password', dest='password', type=str, default=None,
                        help='database password')

    parser.add_argument('--key-prefix',dest='key_prefix',type=str,default="",
                        help='a prefix for sessions, weights and metadata in the database')

    args = parser.parse_args()

    #initialize database & experiment
    try:
        db = Database(host = args.host,port=args.port,password=args.password,
                      default_prefix=args.key_prefix)

        experiment = imp.load_source('loaded_experiment', args.path).make_experiment(db)
    except:
        exc_type, exc, tb = sys.exc_info()

        print_exception(exc_type, exc, tb)
        raise ValueError("The path (%s) should point to a _working_ python module containing function make_experiment "
                         " that can be called with db param only. See breakout.py for example."%(args.path))



    if args.mode == 'play':
        experiment.generate_sessions(args.n_iters, args.batch_size,
                                     reload_period=args.sync_period)
    elif args.mode == 'train':
        experiment.train_on_sessions(args.n_iters,args.batch_size,
                                     save_period=args.sync_period)
    elif args.mode == 'eval':
        experiment.evaluate(1 if np.isinf(args.n_iters) else args.n_iters)
    elif args.mode == 'info':
        raise NotImplementedError('"info" mode not yet implemented')
    elif args.mode == 'clear':
        raise NotImplementedError('"clear" mode not yet implemented')
    else:
        raise ValueError('Please choose mode from the list ("play", "train", "eval", "info", "clear")')





