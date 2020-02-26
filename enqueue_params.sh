#!/bin/sh
# submits set of hyperparameters to file params.csv and adds description in params_readme.txt
# two queues exist: the cluster queue and the parameters queue. 
# for job submission to the cluster I term it enqueue for the cluster, 
# and adding new sets of hyperparameters to params.csv I term it enqueue for the parameters.
# using crontab, the queue status on the cluster is checked to see if additional jobs can be submitted

FILE_NAMES=("tbd_0.pt")
ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations" # should be given in absolute path format
HIDDEN_SIZES=(20 50 100 200)
N_LAYERSS=(2 3)
BIDIRECTIONAL="--bidirectional"
CLASSIFIER="" # CLASSIFIER="--classifier"
IDX_LABELS=(0 1) # set to (0) if classifier to avoid training the same classifier twice
EPOCHS=2000
TRAIN_RATIO=0.8
BATCH_SIZE=128 # 256
N_WORKERS=4
LR=0.001
BETAS="0.9 0.999"
OPT_LEVEL="O2"
SEED=42
LOG_INTERVAL=10

README="testing tbd_0.pt for all parameters"
printf "$(date)\t${README}\n" >> params_readme.txt

for FILE_NAME in ${FILE_NAMES[@]}
do
    for IDX_LABEL in ${IDX_LABELS[@]}
    do
        for HIDDEN_SIZE in ${HIDDEN_SIZES[@]}
        do
            for N_LAYERS in ${N_LAYERSS[@]}
            do
                PARAMS=""
                PARAMS+="${FILE_NAME},"
                PARAMS+="${ROOT_DIR},"
                PARAMS+="${HIDDEN_SIZE},"
                PARAMS+="${N_LAYERS},"
                PARAMS+="${BIDIRECTIONAL},"
                PARAMS+="${CLASSIFIER},"
                PARAMS+="${IDX_LABEL},"
                PARAMS+="${EPOCHS},"
                PARAMS+="${TRAIN_RATIO},"
                PARAMS+="${BATCH_SIZE},"
                PARAMS+="${N_WORKERS},"
                PARAMS+="${LR},"
                PARAMS+="${BETAS},"
                PARAMS+="${OPT_LEVEL},"
                PARAMS+="${SEED},"
                PARAMS+="${LOG_INTERVAL}"
                echo "${PARAMS}" >> params.csv
            done
        done
    done
done
