#! /bin/bash

set -e

STEPS=5000

python eval.py --weightnorm=False --batchnorm=True --batch_size=100 --num_layers=3 --hidden_size=256 --learning_rate=0.25 --max_steps=$STEPS --logdir=logs/highlr_bn --momentum=0.5


python eval.py --weightnorm=True --batchnorm=True --batch_size=100 --num_layers=3 --hidden_size=256 --learning_rate=0.25 --max_steps=$STEPS --logdir=logs/highlr_wnbn --momentum=0.5


python eval.py --weightnorm=True --batchnorm=False --batch_size=100 --num_layers=3 --hidden_size=256 --learning_rate=0.25 --max_steps=$STEPS --logdir=logs/highlr_wn --momentum=0.5


python eval.py --weightnorm=False --batchnorm=False --batch_size=100 --num_layers=3 --hidden_size=256 --learning_rate=0.25 --max_steps=$STEPS --logdir=logs/highlr_plain --momentum=0.5
