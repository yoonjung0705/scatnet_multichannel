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

ROOT_DIR = './data/simulations/two_beads/'
file_name = 'tbd_0.pt'

# Training settings
parser = argparse.ArgumentParser(description='RNN training') # FIXME: change description, read up on argparse
parser.add_argument('--hidden-size', type=int, metavar='N',
                    help='hidden size in LSTM')
parser.add_argument('--num-layers', type=int, metavar='N',
                    help='number of layers in LSTM')
parser.add_argument('--bidirectional', action='store-true', default=False,
                    help='trains bidirectional LSTM')
parser.add_argument('--classifier', action='store_true', default=False,
                    help='train classifier instead of regressor')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--train-ratio', type=float, default=0.8, metavar='TR',
                    help='ratio of training data (default: 0.8)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                    help='number of workers when loading data (default: 1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), metavar='B',
                    help='SGD momentum (default: (0.9, 0.999))')
parser.add_argument('--opt-level', type=str, default='O0', metavar='O',
                    help='optimization level (default: O0)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status (default: 10)')

'''START FIXING FROM HERE'''
file_path = os.path.join(root_dir, file_name + '.pt')
#transformed = 'scat' in file_name
samples = torch.load(file_path)
nums = cu.match_filename(r'{}_meta_rnn_([0-9]+).pt'.format(file_name), root_dir=root_dir)
nums = [int(num) for num in nums]; idx = max(nums) + 1 if nums else 0
file_name_meta = '{}_meta_rnn_{}.pt'.format(file_name, idx)
# data shape: (n_param_1, n_param_2,..., n_param_N, n_samples_total, n_channels, (n_nodes), data_len)
data, labels, label_names = samples['data'], samples['labels'], samples['label_names']
# the number of dimensions that do not correspond to the batch dimension is 4 if scat transformed.
# Otherwise, it's 3
n_none_param_dims = 4 if transformed else 3
n_samples_total = data.shape[-n_none_param_dims]
n_data_total = np.prod(data.shape[:-(n_none_param_dims - 1)])
n_labels = len(label_names) # number of labels to predict
if classifier:
    assert(isinstance(hidden_size, int)), "Invalid format of hidden_size given. Should be type int"
else:
    if n_labels == 1 and isinstance(hidden_size, int): hidden_size = [hidden_size]
    assert(len(hidden_size) == n_labels), "Invalid format of hidden state sizes given.\
        Should have length n_labels"
    assert(all([isinstance(hidden_size_label, int) for hidden_size_label in hidden_size])),\
        "Invalid format of hidden_size given. Should be list with int type elements"

index = nu._train_test_split(n_data_total, train_ratio); index['val'] = index.pop('test')

# reshape data. output is shaped (n_data_total, n_channels * (n_scat_nodes), data_len).
# (n_scat_nodes) means 1 if data not transformed
data = np.reshape(data, (n_data_total, -1, data.shape[-1]))
input_size = data.shape[-2]

# initialize meta data and save it to a file
meta = {'file_name':file_name_meta, 'root_dir':root_dir, 'input_size':input_size,
    'hidden_size':hidden_size, 'n_layers':n_layers, 'bidirectional':bidirectional,
    'classifier':classifier, 'n_epochs_max':n_epochs_max, 'train_ratio':train_ratio,
    'batch_size':batch_size, 'n_workers':n_workers, 'index':index,
    'labels':samples['labels'], 'label_names':samples['label_names']}

labels = np.array(list(product(*labels)), dtype='float32') # shaped (n_conditions, n_labels)
if classifier:
    label_to_idx = {tuple(condition):idx_condition for idx_condition, condition in enumerate(labels)}
    n_conditions = len(label_to_idx)

    meta.update({'epoch':[], 'weights':None, 'elapsed':[], 'loss':{'train':[], 'val':[]},
        'criterion':'cross_entropy_sum', 'label_to_idx':label_to_idx})
    nu._init_meta(**meta)

    labels = np.arange(n_conditions) # shaped (n_conditions,)
    labels = np.repeat(labels, n_samples_total) # shaped (n_conditions * n_samples_total,)
    # which, for example, looks like [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4] 
    # for n_samples_total being 3 and n_conditions being 5

    dataset = nu.TimeSeriesDataset(data, labels, transform=nu.ToTensor())
    # train the neural network for classification
    print("Beginning training of {}:".format(', '.join(samples['label_names'])))
    _train_rnn(dataset, index, hidden_size=hidden_size, n_layers=n_layers,
        bidirectional=bidirectional, classifier=classifier, n_epochs_max=n_epochs_max,
        batch_size=batch_size, n_workers=n_workers, device=device,
        file_name=file_name_meta, root_dir=root_dir, lr=lr, betas=betas)
else:
    meta.update({'epoch':[[] for _ in range(n_labels)], 'weights':[None for _ in range(n_labels)],
        'elapsed':[[] for _ in range(n_labels)], 'loss':[{'train':[], 'val':[]}
            for _ in range(n_labels)], 'criterion':'rmse'})
    nu._init_meta(**meta)
    
    # following is shaped (n_labels, n_conditions)
    labels = labels.swapaxes(0, 1)
    # following is shaped (n_labels, n_data_total)
    labels = np.tile(labels[:, :, np.newaxis], [1, 1, n_samples_total]).reshape([n_labels, n_data_total])
    for idx_label in range(n_labels):
        dataset = nu.TimeSeriesDataset(data, labels[idx_label], transform=nu.ToTensor())
        # train the rnn for the given idx_label
        print("Beginning training of {}:".format(samples['label_names'][idx_label]))
        _train_rnn(dataset, index, hidden_size=hidden_size[idx_label], n_layers=n_layers,
            bidirectional=bidirectional, classifier=classifier, n_epochs_max=n_epochs_max,
            batch_size=batch_size, n_workers=n_workers, device=device, idx_label=idx_label,
            file_name=file_name_meta, root_dir=root_dir, lr=lr, betas=betas)

def _train_rnn(dataset, index, hidden_size, n_layers, bidirectional, classifier, n_epochs_max, batch_size,
        device, n_workers, file_name, root_dir=ROOT_DIR, idx_label=None, lr=0.001, betas=(0.9, 0.999)):
    '''constructs and trains neural networks given the dataloader instance and network structure

    inputs
    ------
    dataset - instance of dataset class inherited from Dataset class
        data should contain keys "data" and "labels"
    index - dict with keys "train" and "val" whose values are list-like indices
    hidden_size - size of hidden state
    n_layers - number of recurrent layers
    bidirectional - if True, becomes a bidirectional LSTM
    classifier: boolean indicating whether it's a classifier or regressor.
    n_epochs_max - maximum number of epochs to run. can be terminated by KeyboardInterrupt
    batch_size - batch size for training
    device - whether to run on gpu or cpu
    n_workers - int indicating number of workers when creating DataLoader instance
    file_name - name of file that contains the empty meta data
    root_dir - root directory of the meta data file
    idx_label - int representing which neural network to train
    lr - float type learning rate
    betas - tuple of floats indicating betas arguments in Adam optimizer

    outputs
    -------
    saves data into given file
    '''
    file_name, _ = os.path.splitext(file_name)
    file_path = os.path.join(root_dir, file_name + '.pt')
    assert(len(dataset) == (len(index['train']) + len(index['val']))),\
        "Size mismatch between dataset and index"
    dataloader = {phase:DataLoader(dataset, sampler=SubsetRandomSampler(index[phase]),
        batch_size=batch_size, num_workers=n_workers) for phase in ['train', 'val']}
    n_data = {phase:len(index[phase]) for phase in ['train', 'val']}

    input_size = dataset[0]['data'].shape[-2]
    meta = torch.load(file_path)
    output_size = len(meta['label_to_idx']) if classifier else 1
    rnn = nu.RNN(input_size, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers,
        bidirectional=bidirectional).to(device)
    optimizer = optim.Adam(rnn.parameters(), lr=lr, betas=betas)
    criterion = nn.CrossEntropyLoss(reduction='sum') if classifier else nn.MSELoss(reduction='sum')
    metric = 'cross_entropy_sum' if classifier else 'rmse'
    time_start = time.time()
    for epoch in range(n_epochs_max):
        try:
            loss_sum = {}
            loss_metric = {}
            for phase in ['train', 'val']:
                rnn.train(phase == 'train')
                loss_sum[phase] = 0.
                for batch in dataloader[phase]:
                    # permute s.t. shape is (data_len, n_data_total, n_channels * (n_scat_nodes))
                    batch_data = batch['data'].permute([2, 0, 1]).to(device)
                    batch_labels = batch['labels']
                    output = rnn(batch_data)
                    # for regression, output of rnn is shaped (batch_size, 1). drop dummy axis
                    if classifier:
                        batch_labels = batch_labels.type(torch.LongTensor)
                    else:
                        output = output[:, 0]
                    batch_labels = batch_labels.to(device)
                    loss = criterion(output, batch_labels)
                    optimizer.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    loss_sum[phase] += loss.data.item()
                # classification: cross entropy sum, regression: RMSE loss per data point
                loss_metric[phase] = loss_sum[phase] if classifier else np.sqrt(loss_sum[phase] / n_data[phase])
            if epoch % 10 == 0:
                time_curr = time.time()
                elapsed = time_curr - time_start
                loss_msg = ("{} out of {} epochs, {}_train:{:.15f}, {}_val:{:.15f}, elapsed seconds:{:.2f}"
                    .format(epoch, n_epochs_max, metric, loss_metric['train'], metric, loss_metric['val'], elapsed))
                print(loss_msg)
                meta = torch.load(file_path)
                if classifier:
                    meta['epoch'].append(epoch)
                    meta['elapsed'].append(elapsed)
                    meta['weights'] = rnn.state_dict()
                    for phase in ['train', 'val']:
                        meta['loss'][phase].append(loss_metric[phase])
                else:
                    meta['epoch'][idx_label].append(epoch)
                    meta['elapsed'][idx_label].append(elapsed)
                    meta['weights'][idx_label] = rnn.state_dict()
                    for phase in ['train', 'val']:
                        meta['loss'][idx_label][phase].append(loss_metric[phase])
                torch.save(meta, file_path)
        except KeyboardInterrupt:
            break

def _train_rnn_cluster(dataset, index, hidden_size, n_layers, bidirectional, classifier, n_epochs_max, batch_size,
        n_workers, file_name, root_dir=ROOT_DIR, idx_label=None, lr=0.001, betas=(0.9, 0.999)):
    '''constructs and trains neural networks given the dataloader instance and network structure for a cluster

    inputs
    ------
    dataset - instance of dataset class inherited from Dataset class
        data should contain keys "data" and "labels"
    index - dict with keys "train" and "val" whose values are list-like indices
    hidden_size - size of hidden state
    n_layers - number of recurrent layers
    bidirectional - if True, becomes a bidirectional LSTM
    classifier: boolean indicating whether it's a classifier or regressor.
    n_epochs_max - maximum number of epochs to run. can be terminated by KeyboardInterrupt
    batch_size - batch size for training
    n_workers - int indicating number of workers when creating DataLoader instance
    file_name - name of file that contains the empty meta data
    root_dir - root directory of the meta data file
    idx_label - int representing which neural network to train
    lr - float type learning rate
    betas - tuple of floats indicating betas arguments in Adam optimizer

    outputs
    -------
    saves data into given file
    '''
    hvd.init() # initialize horovod
    torch.manual_seed(42) # FIXME: necessary here?
    torch.cuda.set_device(hvd.local_rank()) # pin GPU to local rank # FIXME: shouldn't it be .rank() instead of .local_rank()?
    # limit # of CPU threads to be used per worker : FIXME: why do this?
    torch.set_num_threads(4)
    file_name, _ = os.path.splitext(file_name)
    file_path = os.path.join(root_dir, file_name + '.pt')
    if hvd.rank() == 0:
        assert(len(dataset) == (len(index['train']) + len(index['val']))),\
            "Size mismatch between dataset and index"
    # Partition dataset among workers using DistributedSampler
    sampler = {phase:DistributedSampler(Subset(dataset, index[phase]), num_replicas=hvd.size(), rank=hvd.rank()) for phase in ['train', 'val']}
    dataloader = {phase:DataLoader(dataset, sampler=sampler[phase],
        batch_size=batch_size, num_workers=n_workers, pin_memory=True) for phase in ['train', 'val']} 
        # FIXME: pin_memory? perhaps pin_memory means putting data on the gpu, 
    n_data = {phase:len(index[phase]) for phase in ['train', 'val']}

    input_size = dataset[0]['data'].shape[-2]
    meta = torch.load(file_path)
    output_size = len(meta['label_to_idx']) if classifier else 1
    rnn = nu.RNN(input_size, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers,
        bidirectional=bidirectional).cuda()
    optimizer = optim.Adam(rnn.parameters(), lr=lr, betas=betas)
    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=rnn.named_parameters())
    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(rnn.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    # apex
    rnn, optimizer = amp.initialize(rnn, optimizer, opt_level="02")

    criterion = nn.CrossEntropyLoss(reduction='sum') if classifier else nn.MSELoss(reduction='sum')
    criterion = criterion.cuda()
    metric = 'cross_entropy_sum' if classifier else 'rmse'
    time_start = time.time()
    for epoch in range(n_epochs_max):
        try:
            loss_sum = {}
            loss_metric = {}
            for phase in ['train', 'val']:
                rnn.train(phase == 'train')
                loss_sum[phase] = 0.
                for batch in dataloader[phase]:
                    # permute s.t. shape is (data_len, n_data_total, n_channels * (n_scat_nodes))
                    batch_data = batch['data'].permute([2, 0, 1])
                    batch_labels = batch['labels']
                    output = rnn(batch_data)
                    # for regression, output of rnn is shaped (batch_size, 1). drop dummy axis
                    if classifier:
                        batch_labels = batch_labels.type(torch.LongTensor)
                    else:
                        output = output[:, 0]
                    batch_labels = batch_labels
                    loss = criterion(output, batch_labels)
                    optimizer.zero_grad()
                    if phase == 'train':
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                            optimizer.synchronize()
                        with optimizer.skip_synchronize():
                            optimizer.step()
                    loss_sum[phase] += loss.data.item()
                # classification: cross entropy sum, regression: RMSE loss per data point
                loss_metric[phase] = loss_sum[phase] if classifier else np.sqrt(loss_sum[phase] / n_data[phase])
            if epoch % 10 == 0 and hvd.rank() == 0:
                time_curr = time.time()
                elapsed = time_curr - time_start
                loss_msg = ("{} out of {} epochs, {}_train:{:.15f}, {}_val:{:.15f}, elapsed seconds:{:.2f}"
                    .format(epoch, n_epochs_max, metric, loss_metric['train'], metric, loss_metric['val'], elapsed))
                print(loss_msg)
                meta = torch.load(file_path)
                if classifier:
                    meta['epoch'].append(epoch)
                    meta['elapsed'].append(elapsed)
                    meta['weights'] = rnn.state_dict()
                    for phase in ['train', 'val']:
                        meta['loss'][phase].append(loss_metric[phase])
                else:
                    meta['epoch'][idx_label].append(epoch)
                    meta['elapsed'][idx_label].append(elapsed)
                    meta['weights'][idx_label] = rnn.state_dict()
                    for phase in ['train', 'val']:
                        meta['loss'][idx_label][phase].append(loss_metric[phase])
                torch.save(meta, file_path)
        except KeyboardInterrupt:
            break

 
def train_rnn_hvd(file_name, hidden_size, n_layers=1, bidirectional=False, classifier=False, n_epochs_max=2000,
        train_ratio=0.8, batch_size=100, n_workers=4, root_dir=ROOT_DIR, lr=0.001, betas=(0.9, 0.999)):
    '''
    trains the recurrent neural network given a file that contains data.
    this data can be either scat transformed or pure simulated data

    inputs
    ------
    file_name: string type name of file
    hidden_size: list type, sizes of hidden states
    n_layers: number of recurrent layers
    bidirectional: if True, becomes a bidirectional LSTM
    classifier: boolean indicating whether it's a classifier or regressor.
    n_epochs_max: maximum number of epochs to run. 
        can terminate with ctrl + c to move on to next neural network training.
    train_ratio: float indicating ratio for training data. should be between 0 and 1
    batch_size: size of batch for computing gradient
    n_workers: how many subprocesses to use for data loading.
        0 means that the data will be loaded in the main process.
    root_dir: string type root directory name
    lr - float type learning rate
    betas - tuple of floats indicating betas arguments in Adam optimizer

    outputs
    -------
    None: saves weights and meta data into file
    '''
    import horovod.torch as hvd
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    device = hvd.local_rank()
    root_process = hvd.local_rank() == 0

    file_name, _ = os.path.splitext(file_name)
    file_path = os.path.join(root_dir, file_name + '.pt')
    transformed = 'scat' in file_name
    samples = torch.load(file_path)
    nums = cu.match_filename(r'{}_meta_rnn_([0-9]+).pt'.format(file_name), root_dir=root_dir)
    nums = [int(num) for num in nums]; idx = max(nums) + 1 if nums else 0
    file_name_meta = '{}_meta_rnn_{}.pt'.format(file_name, idx)
    # data shape: (n_param_1, n_param_2,..., n_param_N, n_samples_total, n_channels, (n_nodes), data_len)
    data, labels, label_names = samples['data'], samples['labels'], samples['label_names']
    # the number of dimensions that do not correspond to the batch dimension is 4 if scat transformed.
    # Otherwise, it's 3
    n_none_param_dims = 4 if transformed else 3
    n_samples_total = data.shape[-n_none_param_dims]
    n_data_total = np.prod(data.shape[:-(n_none_param_dims - 1)])
    n_labels = len(label_names) # number of labels to predict
    # FIXME: only assert for process 0 or if cluster is False
    if root_process:
        if classifier:
            assert(isinstance(hidden_size, int)), "Invalid format of hidden_size given. Should be type int"
        else:
            if n_labels == 1 and isinstance(hidden_size, int): hidden_size = [hidden_size]
            assert(len(hidden_size) == n_labels), "Invalid format of hidden state sizes given.\
                Should have length n_labels"
            assert(all([isinstance(hidden_size_label, int) for hidden_size_label in hidden_size])),\
                "Invalid format of hidden_size given. Should be list with int type elements"

    index = nu._train_test_split(n_data_total, train_ratio); index['val'] = index.pop('test')

    # reshape data. output is shaped (n_data_total, n_channels * (n_scat_nodes), data_len).
    # (n_scat_nodes) means 1 if data not transformed
    data = np.reshape(data, (n_data_total, -1, data.shape[-1]))
    input_size = data.shape[-2]

    # initialize meta data and save it to a file
    if root_process:
        meta = {'file_name':file_name_meta, 'root_dir':root_dir, 'input_size':input_size,
            'hidden_size':hidden_size, 'n_layers':n_layers, 'bidirectional':bidirectional,
            'classifier':classifier, 'n_epochs_max':n_epochs_max, 'train_ratio':train_ratio,
            'batch_size':batch_size, 'n_workers':n_workers, 'index':index, 'device':device,
            'labels':samples['labels'], 'label_names':samples['label_names']}

    labels = np.array(list(product(*labels)), dtype='float32') # shaped (n_conditions, n_labels)
    if classifier:
        label_to_idx = {tuple(condition):idx_condition for idx_condition, condition in enumerate(labels)}
        n_conditions = len(label_to_idx)

        if root_process:
            meta.update({'epoch':[], 'weights':None, 'elapsed':[], 'loss':{'train':[], 'val':[]},
                'criterion':'cross_entropy_sum', 'label_to_idx':label_to_idx})
            nu._init_meta(**meta)
        labels = np.arange(n_conditions) # shaped (n_conditions,)
        labels = np.repeat(labels, n_samples_total) # shaped (n_conditions * n_samples_total,)
        # which, for example, looks like [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4] 
        # for n_samples_total being 3 and n_conditions being 5

        dataset = nu.TimeSeriesDataset(data, labels, transform=nu.ToTensor())
        # train the neural network for classification
        if root_process: print("Beginning training of {}:".format(', '.join(samples['label_names'])))
        _train_rnn(dataset, index, hidden_size=hidden_size, n_layers=n_layers,
            bidirectional=bidirectional, classifier=classifier, n_epochs_max=n_epochs_max,
            batch_size=batch_size, n_workers=n_workers, device=device,
            file_name=file_name_meta, root_dir=root_dir, lr=lr, betas=betas)
    else:
        if root_process:
            meta.update({'epoch':[[] for _ in range(n_labels)], 'weights':[None for _ in range(n_labels)],
                'elapsed':[[] for _ in range(n_labels)], 'loss':[{'train':[], 'val':[]}
                    for _ in range(n_labels)], 'criterion':'rmse'})
            nu._init_meta(**meta)
        # following is shaped (n_labels, n_conditions)
        labels = labels.swapaxes(0, 1)
        # following is shaped (n_labels, n_data_total)
        labels = np.tile(labels[:, :, np.newaxis], [1, 1, n_samples_total]).reshape([n_labels, n_data_total])
        for idx_label in range(n_labels):
            dataset = nu.TimeSeriesDataset(data, labels[idx_label], transform=nu.ToTensor())
            # train the rnn for the given idx_label
            if root_process: print("Beginning training of {}:".format(samples['label_names'][idx_label]))
            _train_rnn(dataset, index, hidden_size=hidden_size[idx_label], n_layers=n_layers,
                bidirectional=bidirectional, classifier=classifier, n_epochs_max=n_epochs_max,
                batch_size=batch_size, n_workers=n_workers, device=device, idx_label=idx_label,
                file_name=file_name_meta, root_dir=root_dir, lr=lr, betas=betas)

