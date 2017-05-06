#!/bin/bash
module purge
module load cuda
module load cudnn
module load virtualenv
module load pandas/intel/0.17.1
module load scikit-learn/intel/0.18
virtualenv venv-chainer -p python2.7
source venv-chainer/bin/activate
pip install chainer

cd your_project_directory
python train.py $DL_ARGS
