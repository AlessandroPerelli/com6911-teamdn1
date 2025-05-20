#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate machine-learning-312-env

spark-submit main.py