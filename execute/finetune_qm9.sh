#!/bin/bash
########## Instruction ##########
# This script takes three optional environment variables:
# GPU / ADDR / PORT
# e.g. Use gpu 0, 1 and 4 for training, set distributed training
# master address and port to localhost:9901, the command is as follows:
#
# GPU="0,1,4" ADDR=localhost PORT=9901 bash train.sh
#
# Default value: GPU=-1 (use cpu only), ADDR=localhost, PORT=9901
# Note that if your want to run multiple distributed training tasks,
# either the addresses or ports should be different between
# each pair of tasks.
######### end of instruction ##########


########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/../`
echo "Locate the project folder at ${CODE_DIR}"


########## parsing JSON configs ##########
if [ -z $1 ]; then
    echo "Config missing. Usage example: GPU=0,1 bash $0 <config>"
    exit 1;
fi
# CONFIG=`cat $1 | python -c "import sys, json; config = json.load(sys.stdin); config = {k: ' '.join(map(str, v)) if isinstance(v, list) else v for k, v in config.items()}; print(\" \".join([f'--{key}' + (f' {config[key]}' if type(config[key])!=bool else '') for key in config if config[key]]))"`
args=("$@")

echo $CONFIG

GPU="${GPU:--1}" # default using CPU
PROP="${PROP:-homo}" # default using homo


CONFIG=$1
CKPT=$2

export CUDA_VISIBLE_DEVICES=$GPU
GPU_ARR=(`echo $GPU | tr ',' ' '`)
########## start training ##########
cd $CODE_DIR
python train.py --gpus "${!GPU_ARR[@]}" --config ${CONFIG}/train.yaml pretrain_ckpt=${CKPT} dataset.train.property=${PROP} dataset.valid.property=${PROP}


########## evaluation ##########
python -m evaluation.eval_qm9 --config ${CONFIG}/test.yaml --ckpt ./ckpts/QM9 --gpu "${!GPU_ARR[@]}" dataset.train.property=${PROP} dataset.test.property=${PROP}