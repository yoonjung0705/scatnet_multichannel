#!/usr/bin/sh
# submits jobs to the cluster using list of hyperparameters in the params.csv file. In detail, it does the following:
# - updates the status of the jobs by removing lines that were successfully completed (status DONE) for
# files params.csv and exit.log.
# NOTE: it uses the bjobs -a command to see what jobs were successfully finished with the exit status DONE
# However, since this information is only shown for the jobs within the last 1 hour, in order to maintain the parameters list correctly,
# this script must run at least every 1 hour
# - updates the status of the jobs by adding jobids that failed to exit.log
# - goes through line by line in params.csv, submits job if it’s not a failed job until the number of submitted jobs becomes N_JOBS_MAX_NORMAL. update the job id and submission count in params.csv
# - goes through line by line in params.csv, submits job if it’s a failed job until the number of submitted jobs becomes N_JOBS_MAX_EXIT. update the job id and submission count

# two queues exist: the cluster queue and the parameters queue. 
# for job submission to the cluster I term it enqueue for the cluster, 
# and adding new sets of hyperparameters to params.csv I term it enqueue for the parameters.
# cron job is used to run enqueue_cluster.sh periodically

SCATNET_DIR="/nobackup/users/yoonjung/repos/scatnet_multichannel"
cd ${SCATNET_DIR}

# job count
N_JOBS_MAX_NORMAL=5 # max number of jobs allowed to run simutaneously for new jobs. do not go beyond 8
N_JOBS_MAX_EXIT=3 # max number of jobs allowed to run simutaneously for previously failed jobs
SUBMIT_COUNT_MAX=4 # max number of times a job can be submitted to the cluster
BATCH_SIZE_EXIT=64 # use a smaller batch size for previously failed jobs
FILE_NAME_PARAMS="params.csv"
FILE_NAME_JOB="rnn.lsf"
FILE_NAME_JOB_TEMPLATE="rnn_template.lsf"
PAUSE_TIME=60 # wait between job submission to see if jobs can be distributed to different hosts
# the longested time it took for a job to start the actual training was ~180 seconds
# setting time to 60 is expected to make roughly 2~4 jobs being in the same host
# even though a long time is waited between jobs, previous jobs can still get terminated due to new jobs
# so first try to limit the N_JOBS_MAX values, then use this parameter to further decrease the chances of
# jobs being terminated

# delete empty lines. This is due to the usage of the while loop when using the file
# reading is fine with while loop but since we make changes inplace for each line sometimes
# which requires the line number, it's important to have no empty lines
# we delete lines that have 0 or multiple spaces (a line with tabs \t also gets deleted, too)
sed -i '/^\s*$/d' ${FILE_NAME_PARAMS}

# get done jobids. output example: 41422,41423,
JOBIDS_DONE=$(bjobs -a 2> /dev/null | grep "DONE" | cut -d' ' -f1 | awk 'BEGIN { ORS = "," } { print }')
# get exit jobids. output example: 41422\n41423
JOBIDS_EXIT_RECENT=$(bjobs -a 2> /dev/null | grep "EXIT" | cut -d' ' -f1)

# the "No unfinished jobs" is an error message and therefore does not count in wc -l
N_JOBS_SUBMITTED=$(expr $(bjobs 2>/dev/null | wc -l) - 1)
N_JOBS_SUBMITTED=$(( N_JOBS_SUBMITTED > 0 ? N_JOBS_SUBMITTED : 0 ))

# get regular expression of finished jobs for grep. 
# output example for jobs done: 41422|^41423
# the ^ is not added at the beginning to check if JOBIDS_DONE_REGEX is an empty string
JOBIDS_DONE_REGEX="$(echo $JOBIDS_DONE | rev | cut -c 2- | rev | sed 's/,/|^/g')"

# remove the finished jobs in the params.csv file 
if [ ! -z $JOBIDS_DONE_REGEX ]
then
    grep -vE "^$JOBIDS_DONE_REGEX" ${FILE_NAME_PARAMS} > tmp_file
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
            sleep $PAUSE_TIME
        fi
    done

# update number of jobs submitted
N_JOBS_SUBMITTED=$(expr $(bjobs 2>/dev/null | wc -l) - 1)
N_JOBS_SUBMITTED=$(( N_JOBS_SUBMITTED > 0 ? N_JOBS_SUBMITTED : 0 ))

# initialize line count variable for submitting jobs that have failed before
LINE_COUNT=0
# read parameters 1 line at a time. If the number of times the job has been submitted is more than 1, 
# and if the job has been failed before, submit the job and get the id
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

        if [[ $SUBMIT_COUNT -lt $SUBMIT_COUNT_MAX ]] && [[ $SUBMIT_COUNT -ge 1 ]]
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
            sleep $PAUSE_TIME
        fi
    done
