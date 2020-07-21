##
## ATTENTION:
## Before running this training job template/script please read the documentation on bellow URL:
## https://mit-satori.github.io/satori-workload-manager.html#batch-scripts
##
##
## Begin LSF Directives (change only no of required GPUs processes/GPUs and job-name-single name as desired/need)
## - "-n 4" for single AC922, "-n 8" for 2x AC922s, "-n 16" for 4x AC922s etc
##
#BSUB -L /bin/bash
#BSUB -J "rnn"
#BSUB -o "log/rnn_o.%J"
#BSUB -e "log/rnn_e.%J"
#BSUB -n 4
#BSUB -R "span[ptile=4]"
#BSUB -gpu "num=4"
#BSUB -q "normal"


#
# Setup User Environement (Python, WMLCE virtual environment etc)
#
HOME2=/nobackup/users/yoonjung
PYTHON_VIRTUAL_ENVIRONMENT=test
CONDA_ROOT=$HOME2/anaconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
export EGO_TOP=/opt/ibm/spectrumcomputing

# Set up the GPUs and delete any existing scratch directory
cat > setup.sh << EoF_s
#! /bin/sh
##
if [ \${OMPI_COMM_WORLD_LOCAL_RANK} -eq 0 ]
then
  sudo ppc64_cpu --smt=2                 # Set the SMT mode to 2
  ppc64_cpu --smt                        # Verify the SMT mode
  sudo nvidia-smi -rac                   # For POWER9+V100
  sudo nvidia-smi --compute-mode=DEFAULT # Set the compute mode to DEFAULT
  /bin/rm -rf /tmp/data.${USER}          # Delete the scratch directory
fi
EoF_s

#
# Cleaning CUDA_VISIBLE_DEVICES
#  
cat > launch.sh << EoF_s
#! /bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec \$*
EoF_s
chmod +x launch.sh


#
# Runing the training/inference job
# (change only the script name and options after python command)
# wget https://raw.githubusercontent.com/horovod/horovod/master/examples/pytorch_mnist.py  
#
# NOTE: root directory should be given in absolute path format

ddlrun --mpiarg "-x EGO_TOP" "-x HOROVOD_FUSION_THRESHOLD=16777216" -v \
  ./launch.sh python \
  $HOME2/repos/scatnet_multichannel/train_rnn.py --file-name tbd_0.pt --root-dir /nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations \
  --hidden-size 3 --n-layers 2 --bidirectional  \
  --idx-label 0 --epochs 2000 --train-ratio 0.8 --batch-size 128 --n-workers 4 \
  --lr 0.001 --betas 0.9 0.999 --opt-level "O2" --seed 42 --log-interval 10

#
# EoF
#

