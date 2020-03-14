#!/bin/sh
# submits set of hyperparameters to file params.csv and adds description in params.log
# the two leading fields in each line are the job id, number of times the job was submitted
# these two fields get updated when running enqueue_cluster.sh

# two queues exist: the cluster queue and the parameters queue. 
# for job submission to the cluster I term it enqueue for the cluster, 
# and adding new sets of hyperparameters to params.csv I term it enqueue for the parameters.
#FILE_NAMES=("tbd_0.pt" "tbd_1.pt" "tbd_2.pt" "tbd_3.pt" "tbd_4.pt")
#FILE_NAMES=("tbd_0_scat_0.pt" "tbd_0_scat_1.pt" "tbd_1_scat_0.pt" "tbd_1_scat_1.pt" "tbd_2_scat_0.pt" "tbd_2_scat_1.pt" "tbd_3_scat_0.pt" "tbd_3_scat_1.pt" "tbd_4_scat_0.pt" "tbd_4_scat_1.pt")
FILE_NAMES=("data.pt")
#README="training tbd_0,1,2,3,4.pt for hidden size 100 and 2,3 layers parameters"
#README="training scat transformed tbd_0,1,2,3,4.pt for all parameters. scat transform has two versions of hyperparams"
README="training irfp data.pt for all parameters"
#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations" # should be given in absolute path format
ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/experiments/irfp" # should be given in absolute path format
HIDDEN_SIZES=(20 50 100)
#HIDDEN_SIZES=(100)
N_LAYERSS=(2 3)
BIDIRECTIONAL="--bidirectional"
#CLASSIFIER="" # CLASSIFIER="--classifier"
CLASSIFIER="--classifier" # CLASSIFIER="--classifier"
#IDX_LABELS=(1 2) # set to (0) if classifier to avoid training the same classifier twice. Should be (1 2) for two beads 
IDX_LABELS=(0) # set to (0) if classifier to avoid training the same classifier twice. Should be (1 2) for two beads 
EPOCHS=2000
TRAIN_RATIO=0.8
BATCH_SIZE=128 # 256
N_WORKERS=4
LR=0.001
BETAS="0.9 0.999"
OPT_LEVEL="O2"
SEED=42
LOG_INTERVAL=10

printf "$(date)\t${README}\n" >> params.log

for FILE_NAME in ${FILE_NAMES[@]}
do
    for IDX_LABEL in ${IDX_LABELS[@]}
    do
        for HIDDEN_SIZE in ${HIDDEN_SIZES[@]}
        do
            for N_LAYERS in ${N_LAYERSS[@]}
            do
                PARAMS=",0,"
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
