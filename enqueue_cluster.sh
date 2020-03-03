#!/usr/bin/sh
# submits jobs to the cluster using list of hyperparameters in the params.csv file. In detail, it does the following:
# - updates the status of the jobs by removing lines that were successfully completed (status DONE).
# - go through line by line, submit job if it’s not a failed job until the number of submitted jobs becomes N_JOBS_MAX_NORMAL. update the job id and submission count
# - go through line by line, submit job if it’s a failed job until the number of submitted jobs becomes N_JOBS_MAX_EXIT. update the job id and submission count
# 
# two queues exist: the cluster queue and the parameters queue. 
# for job submission to the cluster I term it enqueue for the cluster, 
# and adding new sets of hyperparameters to params.csv I term it enqueue for the parameters.
# cron job is used to run enqueue_cluster.sh periodically

SCATNET_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel"
cd ${SCATNET_DIR}

# job count
N_JOBS_MAX_NORMAL=4
N_JOBS_MAX_EXIT=1
SUBMIT_COUNT_MAX=3
BATCH_SIZE_EXIT=32 # use a smaller batch size for previously failed jobs
FILE_NAME_PARAMS="params.csv"
FILE_NAME_EXIT="exit.log"
FILE_NAME_JOB="rnn.lsf"
FILE_NAME_JOB_TEMPLATE="rnn_template.lsf"

# get done jobids. output example: 41422,41423,
JOBIDS_DONE=$(bjobs -a 2> /dev/null | grep "DONE" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }')
# get exit jobids. output example: 41422\n41423
JOBIDS_EXIT_RECENT=$(bjobs -a 2> /dev/null | grep "EXIT" | cut -d' ' -f1
# TODO: 
# - get the exit jobids from bjobs -a - (A)
# - get the exit jobids from exit.log - (B)
# - for each jobid in jobid_done, see if it's in (B) and if so, remove it from (B)
# - for each jobid in (A), see if it's in (B) and if not, append to (B)
# - the JOBIDS_EXIT_REGEX should be defined from the exit.log file
# - you don't have to define job exit regex with the "|" thing.
# just check with grep directly: grep "$JOBID" -qw "exit.log"

#JOBIDS_EXIT=$(bjobs -a 2> /dev/null | grep "EXIT" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }')
#JOBIDS_EXIT_CUM=$(cat exit.log | paste -sd,)

# the "No unfinished jobs" is an error message and therefore does not count in wc -l
N_JOBS_SUBMITTED=$(expr $(bjobs 2>/dev/null | wc -l) - 1)
N_JOBS_SUBMITTED=$(( N_JOBS_SUBMITTED > 0 ? N_JOBS_SUBMITTED : 0 ))

# get regular expression for grep. 
# output example for jobs done: ^41422|^41423
# output example for jobs exit: 41422|41423
JOBIDS_DONE_REGEX="$(echo $JOBIDS_DONE | rev | cut -c 2- | rev | sed 's/,/|^/g')"
#JOBIDS_EXIT_REGEX="$(echo $JOBIDS_EXIT | rev | cut -c 2- | rev | sed 's/,/|/g')"

# remove the finished jobs in the params.csv file 
if [ ! -z $JOBIDS_DONE_REGEX ]
then
    grep -vE "^$JOBIDS_DONE_REGEX" ${FILE_NAME_PARAMS} > tmp_file
    mv tmp_file ${FILE_NAME_PARAMS}
fi

# remove the finished jobs in the exit.log file 
if [ ! -z $JOBIDS_DONE_REGEX ]
then
    grep -vE "^$JOBIDS_DONE_REGEX" ${FILE_NAME_EXIT} > tmp_file
    mv tmp_file ${FILE_NAME_EXIT}
fi

# add failed jobs in the exit.log file
for JOBID_EXIT_RECENT in $JOBIDS_EXIT_RECENT
do
    if $(! grep -qw $JOBID_EXIT_RECENT ${FILE_NAME_EXIT})
    then
        echo $JOBID_EXIT_RECENT >> ${FILE_NAME_EXIT}
    done
done

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
            cat ${FILE_NAME_JOB_TEMPLATE} | sed \
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
                > ${FILE_NAME_JOB}

            # submit the job and get new job id
            JOBID_NEW=$(bsub < ${FILE_NAME_JOB} | grep -o "[0-9]\+")
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
echo "N_JOBS_MAX_EXIT: $N_JOBS_MAX_EXIT"
echo "checking [ -s FILE_NAME_PARAMS ]"
[ -s ${FILE_NAME_PARAMS} ]
echo $?
# initialize line count variable for submitting jobs that have failed before
LINE_COUNT=0
# read parameters 1 line at a time. If the number of times the job has been submitted is more than 1, 
# and if the job is a recently failed one, submit the job and get the id
# then, replace the jobid and submit count and update the file
cat ${FILE_NAME_PARAMS} | while [[ "${N_JOBS_SUBMITTED}" -lt "${N_JOBS_MAX_EXIT}" ]] && [ -s ${FILE_NAME_PARAMS} ] && IFS=, read \
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
        echo "JOBIDS_EXIT_REGEX:$JOBIDS_EXIT_REGEX"
        echo "SUBMIT_COUNT:$SUBMIT_COUNT"
        echo "SUBMIT_COUNT_MAX:$SUBMIT_COUNT_MAX"
        echo "checking echo JOBID | grep -qEw JOBIDS_EXIT_REGEX"
        $(echo "$JOBID" | grep -qEw "$JOBIDS_EXIT_REGEX")
        echo $?
        echo "checking [[ SUBMIT_COUNT -lt SUBMIT_COUNT_MAX ]]"
        [[ $SUBMIT_COUNT -lt $SUBMIT_COUNT_MAX ]]
        echo $?
        echo "checking [[ SUBMIT_COUNT -ge 1 ]]"
        [[ $SUBMIT_COUNT -ge 1 ]]
        echo $?

        if $(echo "$JOBID" | grep -qw ${FILE_NAME_EXIT}) && [[ $SUBMIT_COUNT -lt $SUBMIT_COUNT_MAX ]] && [[ $SUBMIT_COUNT -ge 1 ]]
        then
            # text substitution using the template
            cat ${FILE_NAME_JOB_TEMPLATE} | sed \
                -e "s/<FILE_NAME>/${FILE_NAME}/g" \
                -e "s|<ROOT_DIR>|${ROOT_DIR}|g" \
                -e "s/<HIDDEN_SIZE>/${HIDDEN_SIZE}/g" \
                -e "s/<N_LAYERS>/${N_LAYERS}/g" \
                -e "s/<BIDIRECTIONAL>/${BIDIRECTIONAL}/g" \
                -e "s/<CLASSIFIER>/${CLASSIFIER}/g" \
                -e "s/<IDX_LABEL>/${IDX_LABEL}/g" \
                -e "s/<EPOCHS>/${EPOCHS}/g" \
                -e "s/<TRAIN_RATIO>/${TRAIN_RATIO}/g" \
                -e "s/<BATCH_SIZE>/${BATCH_SIZE_EXIT}/g" \
                -e "s/<N_WORKERS>/${N_WORKERS}/g" \
                -e "s/<LR>/${LR}/g" \
                -e "s/<BETAS>/${BETAS}/g" \
                -e "s/<OPT_LEVEL>/${OPT_LEVEL}/g" \
                -e "s/<SEED>/${SEED}/g" \
                -e "s/<LOG_INTERVAL>/${LOG_INTERVAL}/g" \
                > ${FILE_NAME_JOB}

            # submit the job and get new job id
            JOBID_NEW=$(bsub < ${FILE_NAME_JOB} | grep -o "[0-9]\+")
            SUBMIT_COUNT_NEW=`expr ${SUBMIT_COUNT} + 1`
            # update the jobid and the submission count in the params.csv file
            awk 'BEGIN { FS=","; OFS="," } NR==line_count { $1=jobid_new; $2=submit_count_new };1' \
                line_count=${LINE_COUNT} jobid_new=${JOBID_NEW} submit_count_new=${SUBMIT_COUNT_NEW} \
                ${FILE_NAME_PARAMS} > tmp_file
            mv tmp_file ${FILE_NAME_PARAMS}
            N_JOBS_SUBMITTED=`expr ${N_JOBS_SUBMITTED} + 1` 
        fi
    done


