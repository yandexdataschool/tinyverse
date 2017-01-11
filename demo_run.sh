#!/bin/bash
bash run_redis.sh
python player.py experiment &
python player.py experiment &
python player.py experiment &
python player.py experiment &
THEANO_FLAGS=device=gpu learner.py experiment
