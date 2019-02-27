import subprocess
import glob
import sys
import os

try:
    folder = sys.argv[1]
    script_folder = sys.argv[2]
except:
    print('Usage: python generate_train_scripts.py path/to/games_folder path/to/scripts_folder')

base_script = """#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
SBATCH --partition=standard
SBATCH --gres=gpu:8
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate textworld

cd baselines/sample_submission_lstm-dqn
"""

os.mkdir(script_folder)

for i, game in enumerate(glob.glob(folder + '*.ulx')):
    with open('{}/script-{}.sh'.format(script_folder, i), 'w') as out_file:
        out_file.write(base_script)
        out_file.write('python train.py --game ../../{}\n'.format(game))
        out_file.write('cd ../..\n')
