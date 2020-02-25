'''standard imports'''
import os
from itertools import product
import numpy as np
import cv2
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd
import time
from apex import amp

'''custom libraries'''
import common_utils as cu
import scat_utils as scu
import net_utils as nu

# regression training example:
# python train_rnn.py --file ./data/simulations/tbd_0.pt --hidden-size 200 --n-layers 2 --bidirectional --idx-label 0 --epochs 2000 --train-ratio 0.8 --batch-size 128 --n-workers 4 --lr 0.001 --betas 0.9 0.999 --opt-level "O2" --seed 42 --log-interval 10

# classifier training example
# python train_rnn.py --file ./data/simulations/tbd_1.pt --hidden-size 200 --n-layers 2 --bidirectional --classifier --epochs 2000 --train-ratio 0.8 --batch-size 128 --n-workers 4 --lr 0.001 --betas 0.9 0.999 --opt-level "O2" --seed 42 --log-interval 10

# most of the arguments can be skipped to use their default values. 
# TODO: create a bash script that iterates over
# file_name_data, hidden_size, n_layers

# Training settings
parser = argparse.ArgumentParser(description='RNN training') # TODO: change description, read up on argparse
parser.add_argument('--file', type=str, metavar='S',
                    help='file path of data to train LSTM')
parser.add_argument('--hidden-size', type=int, metavar='N',
                    help='hidden size in LSTM')
parser.add_argument('--n-layers', type=int, metavar='N',
                    help='number of layers in LSTM')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='trains bidirectional LSTM')
parser.add_argument('--classifier', action='store_true', default=False,
                    help='train classifier instead of regressor')
parser.add_argument('--idx-label', type=int, metavar='N',
                    help='index of label to predict')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--train-ratio', type=float, default=0.8, metavar='TR',
                    help='ratio of training data (default: 0.8)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--n-workers', type=int, default=4, metavar='N',
                    help='number of workers when loading data (default: 1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--betas', nargs='+', default=[0.9, 0.999], metavar='B',
                    help='SGD momentum (default: (0.9, 0.999))')
parser.add_argument('--opt-level', type=str, default='O0', metavar='OL',
                    help='optimization level (default: O0)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status (default: 10)')

args = parser.parse_args()
file_path = args.file
args.file_name = os.path.basename(file_path)
args.root_dir = os.path.dirname(file_path)
if args.classifier: args.idx_label = None
args.betas = tuple(np.array(args.betas, dtype=float))

nu.train_rnn_cluster(file_name=args.file_name, hidden_size=args.hidden_size, n_layers=args.n_layers,
    bidirectional=args.bidirectional, classifier=args.classifier, idx_label=args.idx_label,
    n_epochs_max=args.epochs, train_ratio=args.train_ratio, batch_size=args.batch_size,
    n_workers=args.n_workers, root_dir=args.root_dir, lr=args.lr, betas=args.betas,
    opt_level=args.opt_level, seed=args.seed, log_interval=args.log_interval)
