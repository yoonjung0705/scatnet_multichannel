#!/bin/sh
# submits multiple jobs for the given set of hyperparameters

# constant arguments throughout the loops
# NOTE: root directory should be given in absolute path format!
# This is because functions are run in different modules and the './' is interpreted differently depending on the function
ROOT_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations"
CLASSIFIER=""
#CLASSIFIER="--classifier"
BATCH_SIZE=128

# arguments to loop over
FILE_NAMES=("tbd_0.pt")
IDX_LABELS=(0 1) # set to (0) if classifier to avoid training the same classifier twice
HIDDEN_SIZES=(20 50 100 200)
N_LAYERSS=(2 3)

# job count
N_JOBS_MAX=4
N_JOBS_SUBMITTED=`bjobs | wc -l`
N_JOBS_SUBMITTED=`expr ${N_JOBS_SUBMITTED} - 1`

# pause time
SLEEP_TIME=0

for FILE_NAME in ${FILE_NAMES[@]}
do
    for IDX_LABEL in ${IDX_LABELS[@]}
    do
        for HIDDEN_SIZE in ${HIDDEN_SIZES[@]}
        do
            for N_LAYERS in ${N_LAYERSS[@]}
            do
                if [[ "${N_JOBS_SUBMITTED}" -ge "${N_JOBS_MAX}" ]]
                then
                    break 99
                fi
                cat rnn_template.lsf | sed -e "s|<ROOT_DIR>|${ROOT_DIR}|g" \
                    -e "s/<CLASSIFIER>/${CLASSIFIER}/g" \
                    -e "s/<BATCH_SIZE>/${BATCH_SIZE}/g" \
                    -e "s/<FILE_NAME>/${FILE_NAME}/g" \
                    -e "s/<IDX_LABEL>/${IDX_LABEL}/g" \
                    -e "s/<HIDDEN_SIZE>/${HIDDEN_SIZE}/g" \
                    -e "s/<N_LAYERS>/${N_LAYERS}/g" \
                    > rnn.lsf
                bsub < rnn.lsf

                N_JOBS_SUBMITTED=`expr ${N_JOBS_SUBMITTED} + 1`
                sleep ${SLEEP_TIME}
            done
        done
    done
done
