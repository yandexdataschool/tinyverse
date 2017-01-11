#!/bin/bash
bash run_redis.sh
sleep 30
python player.py experiment &
python player.py experiment &
python player.py experiment &
python player.py experiment &
THEANO_FLAGS=device=gpu python learner.py experiment
