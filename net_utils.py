import os
from itertools import product
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time
import common_utils as cu
import scat_utils as scu

ROOT_DIR = './data/'

class Net(nn.Module):
    def __init__(self, n_nodes):
        super(Net, self).__init__()
        net = []
        for idx in range(len(n_nodes) - 2):
            net.append(nn.Linear(n_nodes[idx], n_nodes[idx + 1]))
            net.append(nn.ELU(inplace=True))
            # nn.BatchNorm1d() can only be processed for batch size being larger than 1.
            # setting to be non-functional for now.
            #net.append(nn.BatchNorm1d(n_nodes[idx + 1]))
            #net.append(nn.Dropout())
        net.append(nn.Linear(n_nodes[-2], n_nodes[-1]))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class RNN(nn.Module):
    '''
    '''
    # FIXME: figure out how to move hidden states h_0 and c_0 to device when doing rnn.to(device)
    # FIXME: check if whether it's necessary to detach() tensors so that unnecessary weight updates
    # do not happen (refer to blog previously visited)
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=False):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional)
        self.h2o = nn.Linear(self.n_directions * hidden_size, output_size)

    def forward(self, input):
        # no need to check input's shape as it'll be checked in self.lstm
        hidden_size = self.hidden_size
        n_layers = self.n_layers
        n_directions = self.n_directions
        batch_size = input.shape[1]
        #h_0 = torch.autograd.Variable(torch.zeros(n_layers, batch_size, hidden_size)) # dtype is default to float32
        #c_0 = nn.Parameter(torch.zeros(n_layers, batch_size, hidden_size))
        #output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        output, (h_n, c_n) = self.lstm(input)
        output = self.h2o(output[-1, :, :])
        
        return output


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        assert(len(data) == len(labels)), "Invalid inputs: shape mismatch between data and labels"
        self._data = data
        self._labels = labels
        self._len = len(self._data)
        self._transform = transform

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        sample = {'data':self._data[index], 'labels':self._labels[index]}
        if self._transform is not None:
            sample = self._transform(sample)

        return sample

class ToTensor:
    def __call__(self, sample):
        dtype = torch.get_default_dtype()
        sample = {'data':torch.tensor(sample['data'], dtype=dtype),
            'labels':torch.tensor(sample['labels'], dtype=dtype)}
        return sample


def _train_test_split(n_data, train_ratio):
    '''
    splits the data indices for training and testing

    inputs
    ------
    n_data: int type number of data
    train_ratio: float indicating ratio for training data. should be between 0 and 1

    outputs
    -------
    index: dict with keys train and test while values being lists of indices
    '''
    assert(train_ratio > 0 and train_ratio < 1), "Invalid train_ratio given. Should be between 0 and 1"
    idx_train = np.random.choice(n_data, int(n_data * train_ratio), replace=False)
    idx_test = np.array(list(set(range(n_data)) - set(idx_train)))
    index = {'train':idx_train, 'test':idx_test}
    return index

def _train_val_test_split(n_data, train_val_ratio):
    '''
    splits the data indices for training and validation and testing

    inputs
    ------
    n_data: int type number of data
    train_val_ratio: list-like with float elements indicating ratio for training and validation data.
        sum should be between 0 and 1

    outputs
    -------
    index: dict with keys train and test while values being lists of indices
    '''
    assert(len(train_val_ratio) == 2), "Split ratio should be given as a length 2 list like input"
    for ratio in train_val_ratio:
        assert(ratio > 0 and ratio < 1), "Invalid train_val_ratio given. Elements should be between 0 and 1"
    assert(sum(train_val_ratio) > 0 and sum(train_val_ratio) < 1), "Invalid train_val_ratio given.\
        Not enough data to be assigned for test purposes"
    idx_train = np.random.choice(n_data, int(n_data * train_val_ratio[0]), replace=False)
    idx_val_test = np.array(list(set(range(n_data)) - set(idx_train)))
    idx_val = np.random.choice(idx_val_test, int(n_data * train_val_ratio[1]), replace=False)
    idx_test = np.array(list(set(idx_val_test) - set(idx_val)))
    index = {'train':idx_train, 'val':idx_val, 'test':idx_test}
    return index

def _init_meta(file_name, root_dir=ROOT_DIR, **kwargs):
    '''initializes the dict type meta data prior to training the neural network'''
    file_name, _ = os.path.splitext(file_name)
    file_path = os.path.join(root_dir, file_name + '.pt')

    meta = {key:value for key, value in kwargs.items()}
    torch.save(meta, file_path)

def train_nn(file_name, n_nodes_hidden, classifier=False, n_epochs_max=2000, train_ratio=0.8, batch_size=100,
        n_workers=4, root_dir=ROOT_DIR, lr=0.001, betas=(0.9, 0.999)):
    '''
    trains the neural network given a file that contains data.
    this data can be either scat transformed or pure simulated data

    inputs
    ------
    file_name: string type name of file
    n_nodes_hidden: list type, where values are nodes (list of nodes) in the hidden layers
        for classification (regression). For regression of multiple labels, the number of lists should match
        with the number of labels to predict
    classifier: boolean indicating whether it's a classifier or regressor.
    n_epochs_max: int, maximum number of epochs to run. 
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

    file_name, _ = os.path.splitext(file_name)
    file_path = os.path.join(root_dir, file_name + '.pt')
    transformed = 'scat' in file_name
    samples = torch.load(file_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nums = cu.match_filename(r'{}_meta_nn_([0-9]+).pt'.format(file_name), root_dir=root_dir)
    nums = [int(num) for num in nums]; idx = max(nums) + 1 if nums else 0
    file_name_meta = '{}_meta_nn_{}.pt'.format(file_name, idx)

    # data shape: (n_param_1, n_param_2,..., n_param_N, n_samples_total, n_channels, (n_nodes), data_len)
    data, labels, label_names = samples['data'], samples['labels'], samples['label_names']
    # the number of dimensions that do not correspond to the batch dimension is 4 if scat transformed.
    # Otherwise, it's 3
    n_none_param_dims = 4 if transformed else 3
    n_samples_total = data.shape[-n_none_param_dims]
    n_data_total = np.prod(data.shape[:-(n_none_param_dims - 1)])
    n_labels = len(label_names) # number of labels to predict
    assert(isinstance(n_nodes_hidden, list)), "Invalid format of nodes given. Should be type list"
    if not classifer:
        if n_labels == 1 and not isinstance(n_nodes_hidden[0], list): n_nodes_hidden = [n_nodes_hidden]
        assert(len(n_nodes_hidden) == n_labels), "Invalid format of nodes given.\
                Should be n_labels number of lists"
        assert(all([isinstance(n_nodes_hidden_label, list) for n_nodes_hidden_label in n_nodes_hidden])),\
            "Invalid format of nodes given. Should provide list of {} lists".format(n_labels)
    index = _train_test_split(n_data_total, train_ratio); index['val'] = index.pop('test')

    # reshape data. output is shaped (n_data_total, n_channels * (n_scat_nodes) * data_len).
    # (n_scat_nodes) means 1 if data not transformed
    data = np.reshape(data, (n_data_total, -1)) 

    # initialize meta data and save it to a file
    meta = {'file_name':file_name_meta, 'root_dir':root_dir, 'n_nodes':n_nodes, 'classifier':classifier,
        'n_epochs_max':n_epochs_max, 'train_ratio':train_ratio, 'batch_size':batch_size,
        'n_workers':n_workers, 'index':index, 'device':device, 'labels':samples['labels'],
        'label_names':samples['label_names']}

    labels = np.array(list(product(*labels)), dtype='float32') # shaped (n_conditions, n_labels)
    if classifier:
        label_to_idx = {tuple(condition):idx_condition for idx_condition, condition in enumerate(labels)}
        n_conditions = len(label_to_idx)
        n_nodes = [data.shape[-1]] + n_nodes_hidden + [n_conditions]
        meta.update({'epoch':[], 'weights':None, 'elapsed':[], 'loss':{'train':[], 'val':[]},
            'criterion':'cross_entropy_sum', 'label_to_idx':label_to_idx, 'n_nodes':n_nodes})
        _init_meta(**meta)
        labels = np.arange(n_conditions) # shaped (n_conditions,)
        labels = np.repeat(labels, n_samples_total) # shaped (n_conditions * n_samples_total,)
        # which, for example, looks like [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4] 
        # for n_samples_total being 3 and n_conditions being 5

        dataset = TimeSeriesDataset(data, labels, transform=ToTensor())
        # train the neural network for classification
        print("Beginning training of {}:".format(', '.join(samples['label_names'])))
        _train_nn(dataset, index, n_nodes_hidden=n_nodes_hidden, classifier=classifier,
            n_epochs_max=n_epochs_max, batch_size=batch_size, device=device, n_workers=n_workers,
            file_name=file_name_meta, root_dir=root_dir)
    else:        
        n_nodes = [[data.shape[-1]] + n_nodes_hidden_label + [1] for n_nodes_hidden_label in n_nodes_hidden]
        meta.update({'epoch':[[] for _ in range(n_labels)], 'weights':[None for _ in range(n_labels)],
            'elapsed':[[] for _ in range(n_labels)], 'loss':[{'train':[], 'val':[]}
                for _ in range(n_labels)], 'criterion':'rmse'})
        _init_meta(**meta)
        # following is shaped (n_labels, n_conditions)
        labels = labels.swapaxes(0, 1)
        # following is shaped (n_labels, n_data_total)
        labels = np.tile(labels[:, :, np.newaxis], [1, 1, n_samples_total]).reshape([n_labels, n_data_total])
        for idx_label in range(n_labels):
            dataset = TimeSeriesDataset(data, labels[idx_label], transform=ToTensor())
            # train the neural network for the given idx_label
            print("Beginning training of {}:".format(samples['label_names'][idx_label]))
            _train_nn(dataset, index, n_nodes_hidden=n_nodes_hidden[idx_label], classifier=classifier,
                n_epochs_max=n_epochs_max, batch_size=batch_size, device=device, n_workers=n_workers,
                idx_label=idx_label, file_name=file_name_meta, root_dir=root_dir)

def _train_nn(dataset, index, n_nodes_hidden, classifier, n_epochs_max, batch_size, device, n_workers,
        file_name, root_dir=ROOT_DIR, idx_label=None, lr=0.001, betas=(0.9, 0.999)):
    '''constructs and trains neural networks given the dataloader instance and network structure

    inputs
    ------
    dataset - instance of dataset class inherited from Dataset class
        data should contain keys "data" and "labels"
    index - dict with keys "train" and "val" whose values are list-like indices
    n_nodes_hidden - list of number of nodes for the hidden layers
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
    n_features = dataset[0]['data'].shape[-1]
    meta = torch.load(file_path)
    output_size = len(meta['label_to_idx']) if classifier else 1
    n_nodes = [n_features] + list(n_nodes_hidden) + [output_size]

    net = Net(n_nodes=n_nodes).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas)
    criterion = nn.CrossEntropyLoss(reduction='sum') if classifier else nn.MSELoss(reduction='sum')
    metric = 'cross_entropy_sum' if classifier else 'rmse'
    time_start = time.time()
    for epoch in range(n_epochs_max):
        try:
            loss_sum = {}
            loss_metric = {}
            for phase in ['train', 'val']:
                net.train(phase == 'train')
                loss_sum[phase] = 0.
                for batch in dataloader[phase]:
                    batch_data, batch_labels = batch['data'].to(device), batch['labels']
                    output = net(batch_data)
                    # for regression, output of nn is shaped (batch_size, 1). drop dummy axis
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
                    meta['weights'] = net.state_dict()
                    for phase in ['train', 'val']:
                        meta['loss'][phase].append(loss_metric[phase])
                else:
                    meta['epoch'][idx_label].append(epoch)
                    meta['elapsed'][idx_label].append(elapsed)
                    meta['weights'][idx_label] = net.state_dict()
                    for phase in ['train', 'val']:
                        meta['loss'][idx_label][phase].append(loss_metric[phase])
                torch.save(meta, file_path)
        except KeyboardInterrupt:
            break

def train_rnn(file_name, hidden_size, n_layers=1, bidirectional=False, classifier=False, n_epochs_max=2000,
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
    file_name, _ = os.path.splitext(file_name)
    file_path = os.path.join(root_dir, file_name + '.pt')
    transformed = 'scat' in file_name
    samples = torch.load(file_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    index = _train_test_split(n_data_total, train_ratio); index['val'] = index.pop('test')

    # reshape data. output is shaped (n_data_total, n_channels * (n_scat_nodes), data_len).
    # (n_scat_nodes) means 1 if data not transformed
    data = np.reshape(data, (n_data_total, -1, data.shape[-1]))
    input_size = data.shape[-2]

    # initialize meta data and save it to a file
    meta = {'file_name':file_name_meta, 'root_dir':root_dir, 'input_size':input_size,
        'hidden_size':hidden_size, 'n_layers':n_layers, 'bidirectional':bidirectional,
        'classifier':classifier, 'n_epochs_max':n_epochs_max, 'train_ratio':train_ratio,
        'batch_size':batch_size, 'n_workers':n_workers, 'index':index, 'device':device,
        'labels':samples['labels'], 'label_names':samples['label_names']}

    labels = np.array(list(product(*labels)), dtype='float32') # shaped (n_conditions, n_labels)
    if classifier:
        label_to_idx = {tuple(condition):idx_condition for idx_condition, condition in enumerate(labels)}
        n_conditions = len(label_to_idx)
        meta.update({'epoch':[], 'weights':None, 'elapsed':[], 'loss':{'train':[], 'val':[]},
            'criterion':'cross_entropy_sum', 'label_to_idx':label_to_idx})
        _init_meta(**meta)
        labels = np.arange(n_conditions) # shaped (n_conditions,)
        labels = np.repeat(labels, n_samples_total) # shaped (n_conditions * n_samples_total,)
        # which, for example, looks like [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4] 
        # for n_samples_total being 3 and n_conditions being 5

        dataset = TimeSeriesDataset(data, labels, transform=ToTensor())
        # train the neural network for classification
        print("Beginning training of {}:".format(', '.join(samples['label_names'])))
        _train_rnn(dataset, index, hidden_size=hidden_size, n_layers=n_layers,
            bidirectional=bidirectional, classifier=classifier, n_epochs_max=n_epochs_max,
            batch_size=batch_size, device=device, n_workers=n_workers,
            file_name=file_name_meta, root_dir=root_dir, lr=lr, betas=betas)
    else:
        meta.update({'epoch':[[] for _ in range(n_labels)], 'weights':[None for _ in range(n_labels)],
            'elapsed':[[] for _ in range(n_labels)], 'loss':[{'train':[], 'val':[]}
                for _ in range(n_labels)], 'criterion':'rmse'})
        _init_meta(**meta)
        # following is shaped (n_labels, n_conditions)
        labels = labels.swapaxes(0, 1)
        # following is shaped (n_labels, n_data_total)
        labels = np.tile(labels[:, :, np.newaxis], [1, 1, n_samples_total]).reshape([n_labels, n_data_total])
        for idx_label in range(n_labels):
            dataset = TimeSeriesDataset(data, labels[idx_label], transform=ToTensor())
            # train the rnn for the given idx_label
            print("Beginning training of {}:".format(samples['label_names'][idx_label]))
            _train_rnn(dataset, index, hidden_size=hidden_size[idx_label], n_layers=n_layers,
                bidirectional=bidirectional, classifier=classifier, n_epochs_max=n_epochs_max,
                batch_size=batch_size, device=device, n_workers=n_workers, idx_label=idx_label,
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
    rnn = RNN(input_size, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers,
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
