#!/bin/bash
#SBATCH -J rnn
#SBATCH -o log/rnn_%j.out
#SBATCH -e log/rnn_%j.err
#SBATCH --mail-user=yoonjung@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4 # if you want 3 nodes where each node has 4 GPUs, you should still write --gres=gpu:4 but --nodes=3
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=23:59:00
###SBATCH --exclusive


#
# Setup User Environement (Python, WMLCE virtual environment etc)
#
HOME2=/nobackup/users/yoonjung
PYTHON_VIRTUAL_ENVIRONMENT=test
CONDA_ROOT=$HOME2/anaconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
#export EGO_TOP=/opt/ibm/spectrumcomputing
ulimit -s unlimited # TODO:?

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

## Horovod execution
horovodrun -np $SLURM_NTASKS -H `cat $NODELIST` python $HOME2/repos/scatnet_multichannel/train_rnn.py \
  --file-name <FILE_NAME> --root-dir <ROOT_DIR> \
  --hidden-size <HIDDEN_SIZE> --n-layers <N_LAYERS> <BIDIRECTIONAL> <CLASSIFIER> \
  --idx-label <IDX_LABEL> --epochs <EPOCHS> --train-ratio <TRAIN_RATIO> \
  --batch-size <BATCH_SIZE> --n-workers <N_WORKERS> \
  --lr <LR> --betas <BETAS> --opt-level <OPT_LEVEL> --seed <SEED> --log-interval <LOG_INTERVAL> <SAVE_MODEL>
