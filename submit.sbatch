#!/bin/bash
# Submission script for use with the xstream cluster
#SBATCH --job-name=polygon_game
#BATCH --output=jobs/polygon_game_%j.txt
#
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres gpu:10

ml pytorch/0.4.0-cp36
source activate torchvis2

THEANO_FLAGS=device=cuda*,floatX=float32 python zoom_game_dqn_pytorch.py
