#!/usr/bin/sh
# submits jobs to the cluster using list of hyperparameters in the params.csv file. In detail, it does the following:
# - updates the status of the jobs by removing lines that were successfully completed (status DONE).
# - go through line by line, submit job if it’s not a failed job until the number of submitted jobs becomes N_JOBS_MAX_NORMAL. update the job id and submission count
# - go through line by line, submit job if it’s a failed job until the number of submitted jobs becomes N_JOBS_MAX_FAILED. update the job id and submission count
# 
# two queues exist: the cluster queue and the parameters queue. 
# for job submission to the cluster I term it enqueue for the cluster, 
# and adding new sets of hyperparameters to params.csv I term it enqueue for the parameters.
# cron job is used to run enqueue_cluster.sh periodically

SCATNET_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel"
cd ${SCATNET_DIR}

# job count
N_JOBS_MAX_NORMAL=4
N_JOBS_MAX_FAILED=1
#JOBID_RUN=$(bjobs -a | grep "RUN" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }' | rev | cut -c 2- | rev )
#JOBID_EXIT=$(bjobs -a | grep "EXIT" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }' | rev | cut -c 2- | rev )
#JOBID_DONE=$(bjobs -a | grep "DONE" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }' | rev | cut -c 2- | rev )

JOBID_RUN=$(bjobs -a | grep "RUN" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }')
JOBID_EXIT=$(bjobs -a | grep "EXIT" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }')
JOBID_DONE=$(bjobs -a | grep "DONE" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }')

# the "No unfinished jobs" is an error message and therefore does not count in wc -l
# FIXME: however there's a header row when there are jobs. So if N_JOBS_SUBMITTED becomes -1, set it to 0
N_JOBS_SUBMITTED=$(expr $(bjobs 2>/dev/null | wc -l) - 1) 
FILE_NAME_PARAMS="params.csv"
SLEEP_TIME=0

# remove jobs that are done
#JOB_DONE_REGEX=$(echo $JOBID_DONE | sed 's/,/|^/g' | cut -c 2-)
JOB_DONE_REGEX=$(echo $JOBID_DONE | rev | cut -c 2- | rev | sed 's/,/|^/g' | cut -c 2-)
grep -E "$JOB_DONE_REGEX" ${FILE_NAME_PARAMS} > ${FILE_NAME_PARAMS}




    # read parameters 1 line at a time
while [[ "${N_JOBS_SUBMITTED_NORMAL}" -lt "${N_JOBS_MAX_NORMAL}" ]] && cat ${FILE_NAME_PARAMS} | IFS=, read \
        JOBID \
        SUBMIT_COUNT \
        FILE_NAME \
        ROOT_DIR \
        HIDDEN_SIZE \
        N_LAYERS \
        BIDIRECTIONAL \
        CLASSIFIER \
        IDX_LABEL \
        EPOCHS \
        TRAIN_RATIO \
        BATCH_SIZE \
        N_WORKERS \
        LR \
        BETAS \
        OPT_LEVEL \
        SEED \
        LOG_INTERVAL
    do
        # text substitution using the template
        cat rnn_template.lsf | sed \
            -e "s/<FILE_NAME>/${FILE_NAME}/g" \
            -e "s|<ROOT_DIR>|${ROOT_DIR}|g" \
            -e "s/<HIDDEN_SIZE>/${HIDDEN_SIZE}/g" \
            -e "s/<N_LAYERS>/${N_LAYERS}/g" \
            -e "s/<BIDIRECTIONAL>/${BIDIRECTIONAL}/g" \
            -e "s/<CLASSIFIER>/${CLASSIFIER}/g" \
            -e "s/<IDX_LABEL>/${IDX_LABEL}/g" \
            -e "s/<EPOCHS>/${EPOCHS}/g" \
            -e "s/<TRAIN_RATIO>/${TRAIN_RATIO}/g" \
            -e "s/<BATCH_SIZE>/${BATCH_SIZE}/g" \
            -e "s/<N_WORKERS>/${N_WORKERS}/g" \
            -e "s/<LR>/${LR}/g" \
            -e "s/<BETAS>/${BETAS}/g" \
            -e "s/<OPT_LEVEL>/${OPT_LEVEL}/g" \
            -e "s/<SEED>/${SEED}/g" \
            -e "s/<LOG_INTERVAL>/${LOG_INTERVAL}/g" \
            > rnn.lsf

        # submit the job
        bsub < rnn.lsf
        sed -i 1d "${FILE_NAME_PARAMS}"
    done
    # the following line should be outside the inner loop for condition check
    N_JOBS_SUBMITTED=`expr ${N_JOBS_SUBMITTED} + 1` 
    sleep ${SLEEP_TIME}
done
