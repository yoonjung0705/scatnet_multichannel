#!/bin/sh
# submits set of hyperparameters to file params.csv and adds description in params.log
# the two leading fields in each line are the job id, number of times the job was submitted
# these two fields get updated when running enqueue_cluster.sh

# two queues exist: the cluster queue and the parameters queue. 
# for job submission to the cluster I term it enqueue for the cluster, 
# and adding new sets of hyperparameters to params.csv I term it enqueue for the parameters.
#############################
#FILE_NAMES=("tbd_0.pt" "tbd_1.pt" "tbd_2.pt" "tbd_3.pt" "tbd_4.pt")
#FILE_NAMES=("tbd_0_scat_0.pt" "tbd_0_scat_1.pt" "tbd_1_scat_0.pt" "tbd_1_scat_1.pt" "tbd_2_scat_0.pt" "tbd_2_scat_1.pt" "tbd_3_scat_0.pt" "tbd_3_scat_1.pt" "tbd_4_scat_0.pt" "tbd_4_scat_1.pt")

#FILE_NAMES=("tbd_0_disp.pt" "tbd_1_disp.pt" "tbd_2_disp.pt" "tbd_3_disp.pt" "tbd_4_disp.pt")
FILE_NAMES=("tbd_0_disp_scat_0.pt" "tbd_0_disp_scat_1.pt" "tbd_1_disp_scat_0.pt" "tbd_1_disp_scat_1.pt" "tbd_2_disp_scat_0.pt" "tbd_2_disp_scat_1.pt" "tbd_3_disp_scat_0.pt" "tbd_3_disp_scat_1.pt" "tbd_4_disp_scat_0.pt" "tbd_4_disp_scat_1.pt")

#FILE_NAMES=("data.pt")
#FILE_NAMES=("data_scat_0.pt" "data_scat_1.pt")

#FILE_NAMES=("data_disp.pt")
#FILE_NAMES=("data_disp_scat_0.pt" "data_disp_scat_1.pt")
#############################
#README="training tbd_0,1,2,3,4.pt (data_len being 2**10=1024, gamma=1-1.5) for hidden size 100,150,200 and 3 layers parameters"
#README="training scat transformed tbd_0,1,2,3,4.pt (data_len being 2**10=1024, gamma=1-1.5) for hidden size 20,50 and 2,3 layers parameters"

#README="training displacement of tbd_0,1,2,3,4.pt (data_len being 2**10=1024, gamma=1-1.5) for hidden size 100,150,200 and 3 layers parameters"
README="training scat transformed displacement of tbd_0,1,2,3,4.pt (data_len being 2**10=1024, gamma=1-1.5) for hidden size 20,50 and 2,3 layers parameters"


#README="training irfp data.pt for hidden size 150,200 and 3 layers"
#README="training scat transformed irfp data.pt for hidden size 20,50 and 2,3 layers"

#README="training displacement of irfp data.pt for hidden size 150,200 and 3 layers"
#README="training scat transformed displacement of irfp data.pt for hidden size 20,50 and 2,3 layers"


#README="training monodisperse 2020_0228 data.pt (data_len being 2**9=512) for hidden size 150,200 and 3 layers"
#README="training scat transformed monodisperse 2020_0228 data.pt (data_len being 2**8=256) for hidden size 20,50 and 2,3 layers"

#README="training displacement of monodisperse 2020_0228 data.pt (data_len being 2**9=512) for hidden size 150,200 and 3 layers"
#README="training scat transformed displacement of monodisperse 2020_0228 data.pt (data_len being 2**8=256) for hidden size 20,50 and 2,3 layers"


#README="training monodisperse 2020_0305 data.pt (data_len being 2**9=512) for hidden size 150,200 and 3 layers"
#README="training scat transformed monodisperse 2020_0305 data.pt (data_len being 2**8=256) for hidden size 20,50 and 2,3 layers"

#README="training displacement of monodisperse 2020_0305 data.pt (data_len being 2**9=512) for hidden size 150,200 and 3 layers"
#README="training scat transformed displacement of monodisperse 2020_0305 data.pt (data_len being 2**8=256) for hidden size 20,50 and 2,3 layers"


#README="training polydisperse 2020_0305 data.pt (data_len being 2**9=512) for hidden size 150,200 and 3 layers"
#README="training scat transformed polydisperse 2020_0305 data.pt (data_len being 2**9=512) for hidden size 20,50 and 2,3 layers"

#README="training displacement of polydisperse 2020_0305 data.pt (data_len being 2**9=512) for hidden size 150,200 and 3 layers"
#README="training scat transformed displacement of polydisperse 2020_0305 data.pt (data_len being 2**9=512) for hidden size 20,50 and 2,3 layers"
#############################
#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations" # should be given in absolute path format
#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations/data_len_2048_gamma_1/pos"
#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations/data_len_2048_gamma_1_1p5/disp"
#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations/data_len_256_gamma_1_1p5/disp"
#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations/data_len_512_gamma_1_1p5/pos"
ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations/data_len_1024_gamma_1_1p5/disp"

#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/experiments/irfp"
#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/experiments/bead/2020_0228/data_len_512/disp"
#ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/experiments/bead/2020_0305/data_len_512_poly/disp"
#############################
HIDDEN_SIZES=(20 50)
#HIDDEN_SIZES=(100 150 200)
N_LAYERSS=(2 3)
#N_LAYERSS=(3)
BIDIRECTIONAL="--bidirectional"

CLASSIFIER="" # regression
#CLASSIFIER="--classifier" # classification

IDX_LABELS=(1 2) # (1 2) for regression. (0 1) for gamma_1
#IDX_LABELS=(0) # set to (0) if classifier to avoid training the same classifier twice. Should be (1 2) for two beads 

EPOCHS=2000
TRAIN_RATIO=0.8
BATCH_SIZE=64 # 256
N_WORKERS=4
LR=0.001
BETAS="0.9 0.999"
OPT_LEVEL="O2"
SEED=42
LOG_INTERVAL=10
SAVE_MODEL=""

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
