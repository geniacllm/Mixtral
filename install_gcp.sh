#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=4:00:00
#$ -j y
#$ -o outputs/
#$ -cwd

set -e

# pip version up
pip install --upgrade pip

# pip install requirements
pip install -r requirements.txt


wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
gunzip -c openmpi-4.1.6.tar.gz | tar xf -
cd ~/moe-recipes/openmpi-4.1.6
./configure --prefix=/usr/local
make all install

cd ~/moe-recipes
# distirbuted training requirements
pip install mpi4py

# huggingface requirements
pip install huggingface_hub

# install flash-atten
pip install ninja packaging wheel
pip install flash-attn==2.3.6 --no-build-isolation

# flash-attn==2.4.2 may require cuda 12.x
