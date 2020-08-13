import os
from itertools import product
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
try:
    import horovod.torch as hvd
    from apex import amp
except:
    pass

import common_utils as cu
import scat_utils as scu

ROOT_DIR = './data/simulations/'

class FCNet(nn.Module):
    def __init__(self, n_nodes):
        super(FCNet, self).__init__()
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


# NOTE: change name to ConvNet or whatever
class Autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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

    def forward(self, input, input_lens=None):
        '''
        inputs
        ------
        input: torch float32 tensor shaped (data_len, batch_size, n_features)
        input_lens: torch long tensor shaped (batch_size,)

        outputs
        -------
        output: torch float32 tensor shaped (batch_size, output_size)
        '''
        # no need to check input's shape as it'll be checked in self.lstm
        hidden_size = self.hidden_size
        n_layers = self.n_layers
        n_directions = self.n_directions
        batch_size = input.shape[1]
        if input_lens is None:
            output, _ = self.lstm(input)
            output = output[-1, :, :]
        elif torch.all(input_lens[0] == input_lens).item(): # if lengths same
            output, _ = self.lstm(input)
            output = output[-1, :, :]
        else: # if lengths different
            data_len = input.shape[0]
            input = pack_padded_sequence(input, input_lens, enforce_sorted=False)
            output, _ = self.lstm(input)
            output, _ = pad_packed_sequence(output, total_length=data_len)
            idx = (input_lens - 1).view(-1, 1).expand(len(input_lens), output.size(2))
            idx = idx.unsqueeze(0)
            output = output.gather(0, idx).squeeze(0)
        #h_0 = torch.autograd.Variable(torch.zeros(n_layers, batch_size, hidden_size)) # dtype is default to float32
        #c_0 = nn.Parameter(torch.zeros(n_layers, batch_size, hidden_size))
        #output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        output = self.h2o(output)
        
        return output


class ImageStackDataset(Dataset):
    def __init__(self, file_name, root_dir, labels=None, transform=None):
        #TODO: if dim is 2, add another dim
        #TODO: consider adding labels in __init__ argument list. if labels is not None, check length. default is None
        file_path = os.path.join(root_dir, file_name)
        self._data = tifffile.imread(file_path)
        #TODO: above line means you read in the whole stack of images in the memory
        # another option: read only the images for the specific batch
        self._labels = labels
        self._len = len(self._data) # provided that 0th dim is number of images
        self._transform = transform
    
    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if self._labels is None:
            sample = {'data':self._data[index], 'labels':None}
        else:
            sample = {'data':self._data[index], 'labels':self._labels[index]}

        if self._transform is not None:
            sample = self._transform(sample)

        return sample


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        assert(len(data) == len(labels)),\
            "Invalid inputs: shape mismatch between data and labels"
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
        data = torch.tensor(sample['data'], dtype=dtype)
        labels = torch.tensor(sample['labels'], dtype=dtype)
        sample_out = {'data': data, 'labels':labels}

        return sample_out


def to_img(x): # this is for after the training
    x = 0.5 * (x + 1) # shifts things towards to the right a bit. num range: (0,1) -> (0,1) but useful in the case where the numbers are slightly outside this range (since you train it it doesn't exactly become (0,1))
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def collate_fn(batch):
    '''
    inputs:
    -------
    batch - list where elements are dictionaries with following keys
        data - n_data length list of ndarrays shaped (n_features, data_len) where data_len varies
        labels - n_data length list of int type elements
    
    outputs:
    --------
    output - dictionary with keys data, labels, input_lens. 
        The values are concatenated versions of the inputs
    '''
    n_data = len(batch)
    n_features = batch[0]['data'].shape[0]
    labels = []
    input_lens = []
    data_out = []
    dtype = torch.get_default_dtype()
    max_data_len = 0
    for idx in range(n_data):
        input_len = batch[idx]['data'].shape[1]
        max_data_len = max(max_data_len, input_len)

    for idx in range(n_data):
        data_slice = batch[idx]['data']
        label = batch[idx]['labels']
        input_len = data_slice.shape[1]

        data_zeros = torch.zeros((n_features, max_data_len - input_len), dtype=dtype)
        data_out.append(torch.cat([data_slice, data_zeros], dim=1))
        labels.append(label)
        input_lens.append(input_len)

    data_out = torch.stack(data_out, dim=0)
    labels = torch.stack(labels, dim=0)
    input_lens = torch.tensor(input_lens).type(torch.LongTensor)
    batch_out = {'data':data_out, 'labels':labels, 'input_lens':input_lens}
    return batch_out

def _train_test_split(n_data, train_ratio, seed=None):
    '''
    splits the data indices for training and testing

    inputs
    ------
    n_data: int type number of data
    train_ratio: float indicating ratio for training data. should be between 0 and 1
    seed - int type seed for random split

    outputs
    -------
    index: dict with keys train and test while values being lists of indices
    '''
    assert(train_ratio > 0 and train_ratio < 1), "Invalid train_ratio given. Should be between 0 and 1"
    if seed is not None: np.random.seed(seed)
    idx_train = np.random.choice(n_data, int(n_data * train_ratio), replace=False)
    idx_test = np.array(list(set(range(n_data)) - set(idx_train)))
    index = {'train':idx_train, 'test':idx_test}
    return index

def _train_val_test_split(n_data, train_val_ratio, seed=None):
    '''
    splits the data indices for training and validation and testing

    inputs
    ------
    n_data: int type number of data
    train_val_ratio: list-like with float elements indicating ratio for training and validation data.
        sum should be between 0 and 1
    seed - int type seed for random split

    outputs
    -------
    index: dict with keys train and test while values being lists of indices
    '''
    assert(len(train_val_ratio) == 2), "Split ratio should be given as a length 2 list like input"
    if seed is not None: np.random.seed(seed)
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

def train_nn(file_name, n_nodes_hidden, classifier=False, n_epochs_max=2000, train_ratio=0.8, batch_size=128,
        n_workers=4, root_dir=ROOT_DIR, lr=0.001, betas=(0.9, 0.999)):
    '''
    trains the neural network given a file that contains data.
    this data can be either scat transformed or pure simulated data

    NOTE: requires refactoring to run on cluster

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
            'criterion':'cross_entropy_mean', 'label_to_idx':label_to_idx, 'n_nodes':n_nodes})
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
    '''constructs and trains neural networks given the dataset instance and network structure
    NOTE: requires refactoring to run on cluster

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

    net = FCNet(n_nodes=n_nodes).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas)
    criterion = nn.CrossEntropyLoss(reduction='sum') if classifier else nn.MSELoss(reduction='sum')
    metric = 'cross_entropy_mean' if classifier else 'rmse'
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
                # classification: cross entropy mean, regression: RMSE loss per data point
                loss_metric[phase] = loss_sum[phase] / n_data[phase] if classifier else np.sqrt(loss_sum[phase] / n_data[phase])
            if epoch % 10 == 0:
                time_curr = time.time()
                elapsed = time_curr - time_start
                loss_msg = ("\t{} out of {} epochs, {}_train:{:.15f}, {}_val:{:.15f}, elapsed seconds:{:.2f}"
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

def train_rnn_noncluster(file_name, hidden_size, n_layers=1, bidirectional=False, classifier=False, idx_label=None, n_epochs_max=2000,
        train_ratio=0.8, batch_size=128, n_workers=4, root_dir=ROOT_DIR, lr=0.001, betas=(0.9, 0.999)):
    '''
    NOTE: to be deprecated. classifier works but regressor has not been integrated yet
    trains the recurrent neural network given a file that contains data.
    this data can be either scat transformed or pure simulated data

    inputs
    ------
    file_name: string type name of file
    hidden_size: list type, sizes of hidden states
    n_layers: number of recurrent layers
    bidirectional: if True, becomes a bidirectional LSTM
    classifier: boolean indicating whether it's a classifier or regressor.
    idx_label: int indicating index of parameter to infer. should be given when classifer is False
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
        assert(idx_label is None), "Invalid idx_label input: should not be given for training classifier"
        assert(isinstance(hidden_size, int)), "Invalid format of hidden_size given. Should be type int"
    else:
        raise NotImplementedError("Training regressor for non-cluster version has not been implemented yet")
        assert(isinstance(idx_label, int)), "Invalid idx_label input: int type idx_label required for training regressor"
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
            'criterion':'cross_entropy_mean', 'label_to_idx':label_to_idx})
        _init_meta(**meta)

        labels = np.arange(n_conditions) # shaped (n_conditions,)
        labels = np.repeat(labels, n_samples_total) # shaped (n_conditions * n_samples_total,)
        # which, for example, looks like [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4] 
        # for n_samples_total being 3 and n_conditions being 5

        dataset = TimeSeriesDataset(data, labels, transform=ToTensor())
        # train the neural network for classification
        print("Beginning training of {}:".format(', '.join(samples['label_names'])))
        _train_rnn_noncluster(dataset, index, hidden_size=hidden_size, n_layers=n_layers,
            bidirectional=bidirectional, classifier=classifier, n_epochs_max=n_epochs_max,
            batch_size=batch_size, n_workers=n_workers, device=device,
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
            _train_rnn_noncluster(dataset, index, hidden_size=hidden_size[idx_label], n_layers=n_layers,
                bidirectional=bidirectional, classifier=classifier, n_epochs_max=n_epochs_max,
                batch_size=batch_size, n_workers=n_workers, device=device, idx_label=idx_label,
                file_name=file_name_meta, root_dir=root_dir, lr=lr, betas=betas)

def _train_rnn_noncluster(dataset, index, hidden_size, n_layers, bidirectional, classifier, n_epochs_max, batch_size,
        device, n_workers, file_name, root_dir=ROOT_DIR, idx_label=None, lr=0.001, betas=(0.9, 0.999)):
    '''constructs and trains neural networks given the dataset instance and network structure

    NOTE: to be deprecated. classifier works but regressor has not been integrated yet

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
    if not classifier:
        raise NotImplementedError("Training regressor for non-cluster version has not been implemented yet")
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
    metric = 'cross_entropy_mean' if classifier else 'rmse'
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
                # classification: cross entropy mean, regression: RMSE loss per data point
                loss_metric[phase] = loss_sum[phase] / n_data[phase] if classifier else np.sqrt(loss_sum[phase] / n_data[phase])
            if epoch % 10 == 0:
                time_curr = time.time()
                elapsed = time_curr - time_start
                loss_msg = ("\t{} out of {} epochs, {}_train:{:.15f}, {}_val:{:.15f}, elapsed seconds:{:.2f}"
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

def train_rnn(file_name, hidden_size, n_layers=1, bidirectional=False, classifier=False, idx_label=None, n_epochs_max=2000,
        train_ratio=0.8, batch_size=128, n_workers=4, root_dir=ROOT_DIR, lr=0.001, betas=(0.9, 0.999),
        opt_level="O0", seed=42, log_interval=10, save_model=False):
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
    idx_label - int representing which neural network to train. should be given only when classifier is False
    n_epochs_max: maximum number of epochs to run. 
        can terminate with ctrl + c to move on to next neural network training.
    train_ratio: float indicating ratio for training data. should be between 0 and 1
    batch_size: size of batch for computing gradient
    n_workers: how many subprocesses to use for data loading.
        0 means that the data will be loaded in the main process.
    root_dir: string type root directory name
    lr - float type learning rate
    betas - tuple of floats indicating betas arguments in Adam optimizer
    opt_level - optimization level
    seed - random seed
    log_interval - how many epochs to wait before logging training status
    save_model - boolean indicating whether to save the model for every log_interval epochs

    outputs
    -------
    None: saves weights and meta data into file
    '''
    # NOTE: regression means you train on data whose parameters are sampled continuously and test also for data whose parameters are sampled continuously, whereas
    # classifier means you train on data on the grid and test on the grid.
    # pass the dataset as an argument to _train_rnn() not with the index but the dataset being a dictionary with keys 'train' and 'val'
    
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    #device = hvd.local_rank()
    root_process = hvd.local_rank() == 0

    file_name, _ = os.path.splitext(file_name)
    file_path = os.path.join(root_dir, file_name + '.pt')
    transformed = 'scat' in file_name
    samples = torch.load(file_path)
    # shape of data: (n_data_total, n_channels, (n_nodes), data_len)
    data, labels, label_names = samples['data'], samples['labels'], samples['label_names']
    n_data_total = len(data)
    if root_process:
        assert(isinstance(hidden_size, int)), "Invalid format of hidden_size given. Should be type int"

    if classifier:
        idx = 0
        if root_process:
            nums = cu.match_filename(r'{}_meta_rnn_([0-9]+).pt'.format(file_name), root_dir=root_dir) # FIXME
            nums = [int(num) for num in nums]; idx = max(nums) + 1 if nums else 0
        idx = hvd.broadcast(torch.tensor(idx), root_rank=0, name='idx_file_meta').item()
        file_name_meta = '{}_meta_rnn_{}.pt'.format(file_name, idx)
    else:
        label = labels[idx_label]
        label_name = label_names[idx_label]
        idx = 0
        if root_process:
            nums = cu.match_filename(r'{}_meta_rnn_([0-9]+)_{}.pt'.format(file_name, label_name), root_dir=root_dir)
            nums = [int(num) for num in nums]; idx = max(nums) + 1 if nums else 0
        idx = hvd.broadcast(torch.tensor(idx), root_rank=0, name='idx_file_meta').item()
        file_name_meta = '{}_meta_rnn_{}_{}.pt'.format(file_name, idx, label_name)

    index = _train_test_split(n_data_total, train_ratio); index['val'] = index.pop('test')

    # reshape data. output is shaped (n_data_total, n_channels * (n_scat_nodes), data_len).
    # (n_scat_nodes) means 1 if data not transformed
    if isinstance(data, np.ndarray):
        data = np.reshape(data, (n_data_total, -1, data.shape[-1]))
    elif isinstance(data, list):
        data = [np.reshape(data_slice, (-1, data_slice.shape[-1])) for data_slice in data]
    else:
        raise ValueError("Invalid type of data given")
    input_size = data[0].shape[0]

    # initialize meta data and save it to a file
    meta = {'file_name':file_name_meta, 'root_dir':root_dir, 'input_size':input_size,
        'hidden_size':hidden_size, 'n_layers':n_layers, 'bidirectional':bidirectional,
        'classifier':classifier, 'n_epochs_max':n_epochs_max, 'train_ratio':train_ratio,
        'batch_size':batch_size, 'n_workers':n_workers, 'index':index,
        'epoch':[], 'elapsed':[], 'loss':{'train':[], 'val':[]},
        'criterion':'cross_entropy_mean' if classifier else 'rmse',
        'labels':labels if classifier else label,
        'label_names':label_names if classifier else label_name}
    if save_model: meta['model'] = []
    if classifier:
        meta['labels_lut'] = samples['labels_lut']
        meta['output_size'] = len(meta['labels_lut'])
        _init_meta(**meta) # done for all processes to ensure data gets loaded after initializing file
        dataset = TimeSeriesDataset(data, labels, transform=ToTensor())
        # train the neural network for classification
        if root_process: print("Training classifier for {}:".format(', '.join(samples['label_names'])))
        _train_rnn(dataset, index, hidden_size=hidden_size, n_layers=n_layers,
            bidirectional=bidirectional, classifier=classifier, n_epochs_max=n_epochs_max,
            batch_size=batch_size, n_workers=n_workers,
            file_name=file_name_meta, root_dir=root_dir, lr=lr, betas=betas,
            opt_level=opt_level, seed=seed, log_interval=log_interval, save_model=save_model)
    else:
        meta['output_size'] = 1
        _init_meta(**meta) # done for all processes to ensure data gets loaded after initializing file
        dataset = TimeSeriesDataset(data, label, transform=ToTensor())
        # train the rnn for the given idx_label
        if root_process: print("Training regressor for {}:".format(label_name))
        _train_rnn(dataset, index, hidden_size=hidden_size, n_layers=n_layers,
            bidirectional=bidirectional, classifier=classifier, n_epochs_max=n_epochs_max,
            batch_size=batch_size, n_workers=n_workers,
            file_name=file_name_meta, root_dir=root_dir, lr=lr, betas=betas,
            opt_level=opt_level, seed=seed, log_interval=log_interval, save_model=save_model)

def _train_rnn(dataset, index, hidden_size, n_layers, bidirectional, classifier, n_epochs_max, batch_size,
        n_workers, file_name, root_dir=ROOT_DIR, lr=0.001, betas=(0.9, 0.999), 
        opt_level="O0", seed=42, log_interval=10, save_model=False):
    '''constructs and trains neural networks given the dataset instance and network structure

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
    lr - float type learning rate
    betas - tuple of floats indicating betas arguments in Adam optimizer
    opt_level - optimization level
    seed - random seed
    log_interval - how many epochs to wait before logging training status
    save_model - boolean indicating whether to save the model

    outputs
    -------
    saves data into given file
    '''
    #hvd.init() # initialize horovod
    torch.manual_seed(seed) # FIXME: necessary here?
    #torch.cuda.set_device(hvd.local_rank()) # pin GPU to local rank
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
        batch_size=batch_size, num_workers=n_workers, collate_fn=collate_fn, pin_memory=True)
        for phase in ['train', 'val']} 
        # FIXME: pin_memory? perhaps pin_memory means putting data on the gpu, 
    n_data = {phase:len(index[phase]) for phase in ['train', 'val']}

    input_size = dataset[0]['data'].shape[-2]
    meta = torch.load(file_path)
    output_size = meta['output_size']
    rnn = RNN(input_size, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers,
        bidirectional=bidirectional).cuda()
    optimizer = optim.Adam(rnn.parameters(), lr=lr, betas=betas)
    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=rnn.named_parameters())
    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(rnn.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    # apex
    rnn, optimizer = amp.initialize(rnn, optimizer, opt_level=opt_level)

    criterion = nn.CrossEntropyLoss(reduction='sum') if classifier else nn.MSELoss(reduction='sum')
    criterion = criterion.cuda()
    metric = 'cross_entropy_mean' if classifier else 'rmse'
    time_start = time.time()
    for epoch in range(n_epochs_max):
        loss_sum = {}
        loss_metric = {}
        for phase in ['train', 'val']:
            rnn.train(phase == 'train')
            loss_sum[phase] = 0.
            for batch in dataloader[phase]:
                # permute s.t. shape is (data_len, n_data_total, n_channels * (n_scat_nodes))
                batch_data = batch['data'].permute([2, 0, 1]).cuda()
                batch_labels = batch['labels'].cuda()
                input_lens = batch['input_lens'].cuda()
                output = rnn(batch_data, input_lens=input_lens)
                # for regression, output of rnn is shaped (batch_size, 1). drop dummy axis
                if classifier:
                    batch_labels = batch_labels.type(torch.cuda.LongTensor)
                else:
                    output = output[:, 0]
                loss = criterion(output, batch_labels)
                optimizer.zero_grad()
                if phase == 'train':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        optimizer.synchronize()
                    with optimizer.skip_synchronize():
                        optimizer.step()
                loss_sum[phase] += loss.data.item()
            # classification: cross entropy mean, regression: RMSE loss per data point
            loss_metric[phase] = loss_sum[phase] / n_data[phase] if classifier else np.sqrt(loss_sum[phase] / n_data[phase])
        if epoch % log_interval == 0 and hvd.rank() == 0:
            time_curr = time.time()
            elapsed = time_curr - time_start
            loss_msg = ("\t{} out of {} epochs, {}_train:{:.15f}, {}_val:{:.15f}, elapsed seconds:{:.2f}"
                .format(epoch, n_epochs_max, metric, loss_metric['train'], metric, loss_metric['val'], elapsed))
            print(loss_msg)
            meta = torch.load(file_path)
            if classifier:
                meta['epoch'].append(epoch)
                meta['elapsed'].append(elapsed)
                if save_model: meta['model'].append(rnn.state_dict())
                for phase in ['train', 'val']:
                    meta['loss'][phase].append(loss_metric[phase])
            else:
                meta['epoch'].append(epoch)
                meta['elapsed'].append(elapsed)
                if save_model: meta['model'].append(rnn.state_dict())
                for phase in ['train', 'val']:
                    meta['loss'][phase].append(loss_metric[phase])
            torch.save(meta, file_path)
