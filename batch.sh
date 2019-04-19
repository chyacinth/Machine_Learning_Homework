#!/bin/bash

python3 ml_hw4/ml_hw4.py -network 784 300 10 -train_size 60000 -test_size 10000 -lr 0.1 -epochs 176 -batch_size 64 -activation sigmoid -loss=square
python3 ml_hw4/ml_hw4.py -network 784 300 10 -train_size 60000 -test_size 10000 -lr 0.1 -epochs 176 -batch_size 64 -activation tanh -loss=square
python3 ml_hw4/ml_hw4.py -network 784 300 10 -train_size 60000 -test_size 10000 -lr 0.1 -epochs 176 -batch_size 64 -activation relu -loss=square
python3 ml_hw4/ml_hw4.py -network 784 300 10 -train_size 60000 -test_size 10000 -lr 0.1 -epochs 176 -batch_size 64 -activation selu -loss=square

python3 ml_hw4/ml_hw4.py -network 784 300 10 -train_size 60000 -test_size 10000 -lr 0.1 -epochs 176 -batch_size 64 -activation sigmoid -loss=cross_entropy
python3 ml_hw4/ml_hw4.py -network 784 300 10 -train_size 60000 -test_size 10000 -lr 0.1 -epochs 176 -batch_size 64 -activation tanh -loss=cross_entropy
python3 ml_hw4/ml_hw4.py -network 784 300 10 -train_size 60000 -test_size 10000 -lr 0.1 -epochs 176 -batch_size 64 -activation relu -loss=cross_entropy
python3 ml_hw4/ml_hw4.py -network 784 300 10 -train_size 60000 -test_size 10000 -lr 0.1 -epochs 176 -batch_size 64 -activation selu -loss=cross_entropy