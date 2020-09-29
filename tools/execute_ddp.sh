#!/bin/bash

MODEL=$1
NUM_GPUS_PER_NODE=$2
NUM_NODES=$3
CONFIG_FILE=$4
WORK_DIR=$5
LEARNING_RATE=$6
EPOCH_NUM=$7
FP16=$8
MASTER_IP=$9
RANK=${10}
#Runs train script through the torch distributed module
NUM_GPUS=$(( NUM_GPUS_PER_NODE * NUM_NODES ))


COMMAND="PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=${NUM_NODES} --node_rank=$RANK --master_addr=${MASTER_IP} --master_port=1234 \
    $(dirname "$0")/train.py ${CONFIG_FILE} --work-dir=${WORK_DIR} --launcher="pytorch" --learning_rate=${LEARNING_RATE} --n_epochs=${EPOCH_NUM} --fp16=${FP16}"


SECONDS=0
eval "${COMMAND}"

#dirty hack to retry if job fails
while [ $? -ne 0 ]; do
    sleep 2
    SECONDS=0
    eval $COMMAND

done
end_time=$SECONDS
TIME=$((end_time))

if [ $RANK -eq 0 ]; then
    echo $TIME > "${WORK_DIR}timetaken"
    echo ""
    echo "Training Complete. Took ${TIME} seconds"
    echo "Pushing metrics to CloudWatch"

    #calling python script to push metrics to cloudwatch
    #make sure to configure aws-cli prior
    DIRECTORY=`dirname $0`
    COMMAND="python ${DIRECTORY}/send_metrics.py ${NUM_GPUS} ${WORK_DIR} ${MODEL} ${EPOCH_NUM}"
    eval $COMMAND

    echo ""
    echo "Metrics pushed. Cleaning work directory.."
    #rm -rf ${WORK_DIR}

    echo "Job Successful"
fi
