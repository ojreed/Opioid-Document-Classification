#!/bin/bash
#
conda create --name testenv python=3.8.10
conda activate testenv
conda install pandas
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c huggingface transformers
conda install scikit-learn
conda install transformers
