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
SUBMIT_COUNT_MAX=3
#JOBID_RUN=$(bjobs -a | grep "RUN" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }' | rev | cut -c 2- | rev )
#JOBID_EXIT=$(bjobs -a | grep "EXIT" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }' | rev | cut -c 2- | rev )
#JOBID_DONE=$(bjobs -a | grep "DONE" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }' | rev | cut -c 2- | rev )

JOBID_RUN=$(bjobs -a | grep "RUN" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }')
JOBID_EXIT=$(bjobs -a | grep "EXIT" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }')
JOBID_DONE=$(bjobs -a | grep "DONE" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }')

# the "No unfinished jobs" is an error message and therefore does not count in wc -l
N_JOBS_SUBMITTED=$(expr $(bjobs 2>/dev/null | wc -l) - 1)
N_JOBS_SUBMITTED=$( N_JOBS_SUBMITTED > 0 ? N_JOBS_SUBMITTED : 0 )
FILE_NAME_PARAMS="params.csv"

# remove jobs that are done
#JOB_DONE_REGEX=$(echo $JOBID_DONE | sed 's/,/|^/g' | cut -c 2-)
JOB_DONE_REGEX=$(echo $JOBID_DONE | rev | cut -c 2- | rev | sed 's/,/|^/g' | cut -c 2-)
grep -E "$JOB_DONE_REGEX" ${FILE_NAME_PARAMS} > ${FILE_NAME_PARAMS}

# initialize line count variable for submitting jobs that have never entered the cluster
LINE_COUNT=0
# read parameters 1 line at a time. If the number of times the job has been submitted is 0, submit the job and get the id
# then, replace the jobid and submit count and update the file
cat ${FILE_NAME_PARAMS} | while [[ "${N_JOBS_SUBMITTED}" -lt "${N_JOBS_MAX_NORMAL}" ]] && [ -s ${FILE_NAME_PARAMS} ] && IFS=, read \
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
        LINE_COUNT=`expr ${LINE_COUNT} + 1`

        if [[ $SUBMIT_COUNT -eq 0 ]]
        then
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

            # submit the job and get new job id
            JOBID_NEW=$(bsub < rnn.lsf > grep -o "<[0-9].*>" | cut -c 2- | rev | cut -c 2- | rev)
            SUBMIT_COUNT_NEW=`expr ${SUBMIT_COUNT} + 1`
            awk 'BEGIN { FS=","; OFS="," } NR==line_count { $2=submit_count };1' line_count=${LINE_COUNT} mytext=${SUBMIT_COUNT} ${FILE_NAME_PARAMS} > ${FILE_NAME_PARAMS}
            N_JOBS_SUBMITTED=`expr ${N_JOBS_SUBMITTED} + 1` 
        fi
    done

# update number of jobs submitted
N_JOBS_SUBMITTED=$(expr $(bjobs 2>/dev/null | wc -l) - 1)
N_JOBS_SUBMITTED=$( N_JOBS_SUBMITTED > 0 ? N_JOBS_SUBMITTED : 0 )

# initialize line count variable for submitting jobs that have failed before
LINE_COUNT=0
# read parameters 1 line at a time. If the number of times the job has been submitted is more than 1, submit the job and get the id
# then, replace the jobid and submit count and update the file
cat ${FILE_NAME_PARAMS} | while [[ "${N_JOBS_SUBMITTED}" -lt "${N_JOBS_MAX_FAILED}" ]] && [ -s ${FILE_NAME_PARAMS} ] && IFS=, read \
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
        LINE_COUNT=`expr ${LINE_COUNT} + 1`

        if [[ $SUBMIT_COUNT -lt $SUBMIT_COUNT_MAX ]] && [[ $SUBMIT_COUNT -ge 1 ]]
        then
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

            # submit the job and get new job id
            JOBID_NEW=$(bsub < rnn.lsf > grep -o "<[0-9].*>" | cut -c 2- | rev | cut -c 2- | rev)
            SUBMIT_COUNT_NEW=`expr ${SUBMIT_COUNT} + 1`
            awk 'BEGIN { FS=","; OFS="," } NR==line_count { $2=submit_count };1' line_count=${LINE_COUNT} mytext=${SUBMIT_COUNT} ${FILE_NAME_PARAMS} > ${FILE_NAME_PARAMS}
            N_JOBS_SUBMITTED=`expr ${N_JOBS_SUBMITTED} + 1` 
        fi
    done



