#!/bin/bash
set -ex

# Script to facilitate multi-node training within AWS for
# models supported in mmdetection

check_config() {
  #Checking if entered backbone is supported by the entered model
 
  supported_configs=($(ls /shared/mmdetection/configs/${MODEL}))
  delete=README.md
  supported_configs=( "${supported_configs[@]/$delete}" )

  if [[ " ${supported_configs[*]} " != *"$CONFIG"* ]];
  then
    echo "Unsupported config entered. Please enter one of the following configs for ${MODEL}"
    echo "${supported_configs[*]}"
    exit 1
  else
    CONFIG_FILE="/shared/mmdetection/configs/${MODEL}/${CONFIG}"
  fi
}

check_args() {
#Checking number of GPU's and nodes available, 
#and if model entered is supported

if [ -z "${DDP}" ]; then
    read NUM_AVAILABLE_NODES <<< $(sinfo | awk 'FNR == 2 { print $4 }')
    echo "Computing number of GPU's available"
    read NUM_GPUS_PER_NODE <<< $(srun -N 1 nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)

    NUM_GPUS_AVAILABLE=$(($NUM_AVAILABLE_NODES * $NUM_GPUS_PER_NODE))

    if (($gpu_num > $NUM_GPUS_AVAILABLE)); then
        echo "Only ${NUM_GPUS_AVAILABLE} GPU's available. Please try again"
        exit 1
    fi
fi

    if [[ " ${supported_models[*]} " != *"$MODEL"* ]];
    then
        echo "Unsupported model entered."
        echo "Please enter one of the following models"
        echo "${supported_models[*]}"
        exit 1
fi
}

supported_models=($(ls /shared/mmdetection/configs/))
delete=_base_
supported_models=( "${supported_models[@]/$delete}" )

usage="$(basename "$0") -m model_name -c config_file -g gpu_num -- program to run training jobs within mmdetection for single/multiple nodes

where:
    config should be filename within the model config directory
    gpu_num sets  number of GPU's.
    -l sets learning rate (optional)
    -n sets num_epochs (optional)
    -f enable mixed precision training (0/1)(optional)
    -r node rank (if specified, will run using torch.distributed instead of slurm)
    -i master ip address (if node rank is specified, this must be specified as well)
    -o number of nodes (optional)
    Supported model names are ${supported_models[@]}

    "


while getopts ":m:c:g:l:n:f:r:i:o:" opt; do
  case $opt in
    m) MODEL="$OPTARG"
    ;;
    c) CONFIG="$OPTARG"
    ;;
    g) gpu_num="$OPTARG"
    ;;
    l) LEARNING_RATE="$OPTARG"
    ;;
    n) N_EPOCHS="$OPTARG"
    ;;
    f) FP16="$OPTARG"
    ;;
    r) RANK="$OPTARG"
    ;;
    i) MASTER_IP="$OPTARG"
    ;;
    o) NUM_NODES="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2; echo "$usage"; exit 1
    ;;
    *) echo $usage; exit;;
  esac
done

if [ -z "${MODEL}" ]; then
    echo "Please specify model with -m"
    echo "${usage}"
    exit 1
fi
if [ -z "${CONFIG}" ]; then
    echo "Please specify config file with -f"
    echo "${usage}"
    exit 1
fi
if [ -z "${gpu_num}" ]; then
    echo "Please specify number of GPU's to train with -g"
    echo "${usage}"
    exit 1
fi

if [ -z "${RANK}" ] && [ -z "${MASTER_IP}" ]; then
    echo "Distributing training with Slurm"
else
   if [ -z ${RANK} ]; then
       echo "If -i option is specified, -r option has to be specified as well."
       exit 1
   elif [ -z ${MASTER_IP} ]; then
       echo "If -r option is specified, -i option has to be specified as well."
       exit 1
   else
       echo "Distributing training with torch.distributed module"
       DDP=1
   fi
   
fi


check_args
check_config

set -x


NUM_NODES=${NUM_NODES:-1}
#Directory to save logs
WORK_DIR="work_dir_${NUM_NODES}_nodes_${MODEL}/"

SRUN_ARGS=${SRUN_ARGS:-""}
#lr calculated based on number of GPUs
lr=$(echo "scale=2;0.01*${gpu_num}/4" | bc)

LEARNING_RATE=${LEARNING_RATE:-$lr}
echo $LEARNING_RATE
LEARNING_RATE=$(echo "scale=2;0.01*${gpu_num}/4" | bc)

#Setting cap on learning rate-does not converge above 0.08
if (( $(echo "$LEARNING_RATE > 0.08" |bc -l) )); then
  LEARNING_RATE=.08
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
N_EPOCHS=${N_EPOCHS:-12}
FP16=${FP16:-0}
DIRECTORY=`dirname $0`


echo "Printing rank"
echo ${RANK}
if [ ! -z ${DDP} ]; then
    COMMAND="${DIRECTORY}/execute_ddp.sh $MODEL 8 $NUM_NODES $CONFIG_FILE $WORK_DIR $LEARNING_RATE $N_EPOCHS $FP16 $MASTER_IP $RANK"
else
    COMMAND="${DIRECTORY}/execute_srun.sh $MODEL $NUM_GPUS_PER_NODE $gpu_num $CONFIG_FILE $WORK_DIR $LEARNING_RATE $N_EPOCHS $FP16"
fi

eval bash $COMMAND


