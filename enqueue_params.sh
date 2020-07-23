#!/bin/sh
# submits set of hyperparameters to file params.csv and adds description in params.log
# the two leading fields in each line are the job id, number of times the job was submitted
# these two fields get updated when running enqueue_cluster.sh

# two queues exist: the cluster queue and the parameters queue. 
# for job submission to the cluster I term it enqueue for the cluster, 
# and adding new sets of hyperparameters to params.csv I term it enqueue for the parameters.
#############################
#FILE_NAMES=("tbd_4.pt")
#FILE_NAMES=("tbd_4_scat_0.pt")
#FILE_NAMES=("tbd_0.pt" "tbd_1.pt" "tbd_2.pt" "tbd_3.pt")
#FILE_NAMES=("tbd_5.pt" "tbd_6.pt" "tbd_7.pt" "tbd_8.pt")
#FILE_NAMES=("tbd_0_scat_0.pt" "tbd_1_scat_0.pt" "tbd_2_scat_0.pt" "tbd_3_scat_0.pt")
#FILE_NAMES=("tbd_0_scat_0.pt" "tbd_0_scat_1.pt" "tbd_1_scat_0.pt" "tbd_1_scat_1.pt" "tbd_2_scat_0.pt" "tbd_2_scat_1.pt" "tbd_3_scat_0.pt" "tbd_3_scat_1.pt" "tbd_4_scat_0.pt" "tbd_4_scat_1.pt")
#FILE_NAMES=("tbd_5_scat_0.pt" "tbd_5_scat_1.pt" "tbd_6_scat_0.pt" "tbd_6_scat_1.pt" "tbd_7_scat_0.pt" "tbd_7_scat_1.pt" "tbd_8_scat_0.pt" "tbd_8_scat_1.pt")


#FILE_NAMES=("data.pt")
FILE_NAMES=("data_scat_0.pt")

#############################
#README="training tbd_4.pt (data_len being 2**11=2048, gamma=1-3) for hidden size 50,100,150,200,250 and 3 layers parameters"
#README="training scat transformed tbd_4.pt (data_len being 2**11=2048, gamma=1-3) for hidden size 20,50,100,150,200,250 and 3 layers parameters"
#README="training tbd_0,1,2,3.pt (data_len being 2**7=128, gamma=1-3) for hidden size 20,50,100,150,200,250 and 3 layers parameters for new k_ratios and diff_coef_ratios data"
#README="training tbd_5,6,7,8.pt (data_len being 2**8=256, gamma=1-1.5) for hidden size 50,100,150,200,250,300 and 3 layers parameters"
#README="training scat transformed tbd_0,1,2,3.pt (data_len being 2**7=128, gamma=1-3) for hidden size 20,50,100,150,200 and 3 layers parameters"
#README="training scat transformed tbd_0,1,2,3.pt (data_len being 2**7=128, gamma=1-3) for hidden size 20,50,100,150,200 and 3 layers parameters for new k_ratios and diff_coef_ratios data"
#README="training scat transformed tbd_0,1,2,3,4.pt (data_len being 2**10=1024, gamma=1-1.5) for hidden size 20,50 and 2,3 layers parameters"
#README="training scat transformed tbd_5,6,7,8.pt (data_len being 2**8=256, gamma=1-1.5) for hidden size 20,50,100,150,200,250 and 3 layers parameters"



#README="training irfp data.pt for hidden size 150,200 and 3 layers"
#README="training scat transformed irfp data.pt for hidden size 150 and 3 layers"



#README="training monodisperse 2020_0228 data.pt (data_len being 2**9=512) for hidden size 150,200 and 3 layers"
#README="training scat transformed monodisperse 2020_0228 data.pt (data_len being 2**8=256) for hidden size 20,50 and 2,3 layers"



#README="training monodisperse 2020_0305 data.pt (data_len being 2**9=512) for hidden size 150,200 and 3 layers"
#README="training scat transformed monodisperse 2020_0305 data.pt (data_len being 2**8=256) for hidden size 20,50 and 2,3 layers"



#README="training polydisperse 2020_0305 data.pt (data_len being 2**9=512, train_val tracks = 702) for hidden size 50,100,150,200,250 and 3 layers"
#README="training scat transformed polydisperse 2020_0305 data.pt (data_len being 2**9=512, train_val tracks = 702) for hidden size 20,50,100,150,200,250 and 3 layers"

#README="training polydisperse 2020_0319 data.pt (data_len being 2**10=1024, train_val tracks = 29) for hidden size 50,100,150,200,250 and 3 layers"
#README="training scat transformed polydisperse 2020_0319 data.pt (data_len being 2**10=1024, train_val tracks = 29) for hidden size 20,50,100,150,200,250 and 3 layers"

README=""
#############################
# should be given in absolute path format
#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations/data_len_2048_gamma_1_3_k_1_7_t_4_10/"
ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/experiments/bead/2020_0319/data_len_256_train_val_175_test_59_sep_2"
#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/experiments/bead/2020_0319/data_len_1024_train_val_26_test_26_sep"
#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/experiments/irfp"
#############################
HIDDEN_SIZES=(50)
#HIDDEN_SIZES=(50 100 150 200 250)
N_LAYERSS=(2)
BIDIRECTIONAL="--bidirectional"

#CLASSIFIER="" # regression
CLASSIFIER="--classifier" # classification

#IDX_LABELS=(1 2) # (1 2) for two beads
IDX_LABELS=(0) # (0) for classification

EPOCHS=10000
TRAIN_RATIO=0.8
BATCH_SIZE=64
N_WORKERS=0
LR=0.001
BETAS="0.9 0.999"
OPT_LEVEL="O2"
SEED=42
LOG_INTERVAL=50
#SAVE_MODEL=""
SAVE_MODEL="--save-model"

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
                PARAMS+="${LOG_INTERVAL},"
                PARAMS+="${SAVE_MODEL}"
                echo "${PARAMS}" >> params.csv
            done
        done
    done
done
