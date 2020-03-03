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
BATCH_SIZE_FAILED=32 # use a smaller batch size for previously failed jobs

# get jobids. output example: 41422,41423,
JOBID_DONE=$(bjobs -a 2> /dev/null | grep "DONE" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }')
# TODO: 
# - get the exit jobids from bjobs -a - (A)
# - get the exit jobids from fail.log - (B)
# - for each jobid in (A), see if it's in (B) and if not, append to (B)
# - for each jobid in jobid_done, see if it's in (B) and if so, remove it from (B)
# - the JOB_EXIT_REGEX should be defined from the fail.log file
# - you don't have to define job exit regex with the "|" thing.
# just check with grep directly: grep "$JOBID" -qw "fail.log"

#JOBID_EXIT=$(bjobs -a 2> /dev/null | grep "EXIT" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }')
#JOBID_EXIT_CUM=$(cat fail.log | paste -sd,)

# the "No unfinished jobs" is an error message and therefore does not count in wc -l
N_JOBS_SUBMITTED=$(expr $(bjobs 2>/dev/null | wc -l) - 1)
N_JOBS_SUBMITTED=$(( N_JOBS_SUBMITTED > 0 ? N_JOBS_SUBMITTED : 0 ))
FILE_NAME_PARAMS="params.csv"

# get regular expression for grep. 
# output example for jobs done: ^41422|^41423
# output example for jobs exit: 41422|41423
JOB_DONE_REGEX="$(echo $JOBID_DONE | rev | cut -c 2- | rev | sed 's/,/|^/g')"
JOB_EXIT_REGEX="$(echo $JOBID_EXIT | rev | cut -c 2- | rev | sed 's/,/|/g')"

if [ ! -z $JOB_DONE_REGEX ]
then
    grep -vE "^$JOB_DONE_REGEX" ${FILE_NAME_PARAMS} > tmp_file
    mv tmp_file ${FILE_NAME_PARAMS}
fi



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
            JOBID_NEW=$(bsub < rnn.lsf | grep -o "[0-9]\+")
            SUBMIT_COUNT_NEW=`expr ${SUBMIT_COUNT} + 1`
            # update the jobid and the submission count in the params.csv file
            awk 'BEGIN { FS=","; OFS="," } NR==line_count { $1=jobid_new; $2=submit_count_new };1' \
                line_count=${LINE_COUNT} jobid_new=${JOBID_NEW} submit_count_new=${SUBMIT_COUNT_NEW} \
                ${FILE_NAME_PARAMS} > tmp_file
            mv tmp_file ${FILE_NAME_PARAMS}
            N_JOBS_SUBMITTED=`expr ${N_JOBS_SUBMITTED} + 1` 
        fi
    done

# update number of jobs submitted
N_JOBS_SUBMITTED=$(expr $(bjobs 2>/dev/null | wc -l) - 1)
N_JOBS_SUBMITTED=$(( N_JOBS_SUBMITTED > 0 ? N_JOBS_SUBMITTED : 0 ))

echo "N_JOBS_SUBMITTED: $N_JOBS_SUBMITTED"
echo "N_JOBS_MAX_FAILED: $N_JOBS_MAX_FAILED"
echo "checking [ -s FILE_NAME_PARAMS ]"
[ -s ${FILE_NAME_PARAMS} ]
echo $?
# initialize line count variable for submitting jobs that have failed before
LINE_COUNT=0
# read parameters 1 line at a time. If the number of times the job has been submitted is more than 1, 
# and if the job is a recently failed one, submit the job and get the id
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

        echo "LINE_COUNT:$LINE_COUNT"
        echo "JOBID:$JOBID"
        echo "JOB_EXIT_REGEX:$JOB_EXIT_REGEX"
        echo "SUBMIT_COUNT:$SUBMIT_COUNT"
        echo "SUBMIT_COUNT_MAX:$SUBMIT_COUNT_MAX"
        echo "checking echo JOBID | grep -qEw JOB_EXIT_REGEX"
        $(echo "$JOBID" | grep -qEw "$JOB_EXIT_REGEX")
        echo $?
        echo "checking [[ SUBMIT_COUNT -lt SUBMIT_COUNT_MAX ]]"
        [[ $SUBMIT_COUNT -lt $SUBMIT_COUNT_MAX ]]
        echo $?
        echo "checking [[ SUBMIT_COUNT -ge 1 ]]"
        [[ $SUBMIT_COUNT -ge 1 ]]
        echo $?

        if $(echo "$JOBID" | grep -qEw "$JOB_EXIT_REGEX") && [[ $SUBMIT_COUNT -lt $SUBMIT_COUNT_MAX ]] && [[ $SUBMIT_COUNT -ge 1 ]]
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
                -e "s/<BATCH_SIZE>/${BATCH_SIZE_FAILED}/g" \
                -e "s/<N_WORKERS>/${N_WORKERS}/g" \
                -e "s/<LR>/${LR}/g" \
                -e "s/<BETAS>/${BETAS}/g" \
                -e "s/<OPT_LEVEL>/${OPT_LEVEL}/g" \
                -e "s/<SEED>/${SEED}/g" \
                -e "s/<LOG_INTERVAL>/${LOG_INTERVAL}/g" \
                > rnn.lsf

            # submit the job and get new job id
            JOBID_NEW=$(bsub < rnn.lsf | grep -o "[0-9]\+")
            SUBMIT_COUNT_NEW=`expr ${SUBMIT_COUNT} + 1`
            # update the jobid and the submission count in the params.csv file
            awk 'BEGIN { FS=","; OFS="," } NR==line_count { $1=jobid_new; $2=submit_count_new };1' \
                line_count=${LINE_COUNT} jobid_new=${JOBID_NEW} submit_count_new=${SUBMIT_COUNT_NEW} \
                ${FILE_NAME_PARAMS} > tmp_file
            mv tmp_file ${FILE_NAME_PARAMS}
            N_JOBS_SUBMITTED=`expr ${N_JOBS_SUBMITTED} + 1` 
        fi
    done


