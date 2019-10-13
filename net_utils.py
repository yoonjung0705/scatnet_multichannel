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
    NOTE: this class should be general enough so that any input can be used
    either the raw time series or the scattering transform result.
    also, should allow variable length
    '''
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=False):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # no need to check input's shape as it'll be checked in self.lstm
        hidden_size = self.hidden_size
        n_layers = self.n_layers
        n_directions = self.n_directions
        batch_size = input.shape[1]
        h_0 = torch.zeros(n_layers, batch_size, hidden_size) # dtype is default to float32
        c_0 = torch.zeros(n_layers, batch_size, hidden_size)

        output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        output = self.h2o(output[-1, :, :])
        
        return output

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss(reduction='sum')
rnn = RNN(input_size, hidden_size, output_size, n_layers=n_layers, bidirectional=bidirectional)
output = rnn(input)
loss = criterion(output, target)
loss.backward()


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
        sample = {'data':self._data[index, :], 'labels':self._labels[index]}
        if self._transform is not None:
            sample = self._transform(sample)

        return sample

class ToTensor:
    def __call__(self, sample):
        dtype = torch.get_default_dtype()
        sample = {'data':torch.tensor(sample['data'], dtype=dtype),
            'labels':torch.tensor(sample['labels'], dtype=dtype)}
        return sample


def train(file_name, n_nodes_hidden, avg_len, log_scat=True, n_filter_octave=[1, 1], n_epochs_max=2000, train_ratio=0.8, batch_size=100, n_workers=4, root_dir=ROOT_DIR):
    file_name, _ = os.path.splitext(file_name)
    file_path = os.path.join(root_dir, file_name + '.pt')
    samples = torch.load(file_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nums = cu.match_filename(r'{}_meta_nn_([0-9]+).pt'.format(file_name), root_dir=root_dir)
    nums = [int(num) for num in nums]
    idx = max(nums) + 1 if nums else 0
    file_name_meta = '{}_meta_nn_{}.pt'.format(file_name, idx)
    file_path_meta = os.path.join(root_dir, file_name_meta)

    data, labels, label_names = samples['data'], samples['labels'], samples['label_names']
    n_conditions = np.prod(data.shape[:-3])
    n_samples_total, n_channels, data_len = data.shape[-3:]
    n_labels = len(label_names) # number of labels to predict
    phases = ['train', 'val']
    n_samples = {'train':int(n_samples_total * train_ratio)}
    n_samples['val'] = n_samples_total - n_samples['train']
    idx_train = np.random.choice(n_samples_total, n_samples['train'], replace=False)
    idx_val = np.array(list(set(range(n_samples_total)) - set(idx_train)))
    index = {'train':idx_train, 'val':idx_val}
    n_data = {phase:n_samples[phase] * n_conditions for phase in phases}
    n_data_total = n_data['train'] + n_data['val']

    # flatten the data to shape (n_data_total, n_channels, data_len)
    data = np.reshape(data, [-1, n_channels, data_len])
    # perform scattering transform
    scat = scu.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
    S = scat.transform(data)
    if log_scat: S = scu.log_scat(S)
    S = scu.stack_scat(S) # shaped (n_data_total, n_channels, n_scat_nodes, data_len)
    data_scat = np.reshape(S, (n_data_total, -1)) # shaped (n_data_total, n_channels * n_scat_nodes * data_len)

    n_nodes_input_layer = data_scat.shape[-1]
    n_nodes = [n_nodes_input_layer] + list(n_nodes_hidden) + [1]
    # initialize meta data and save it to a file
    _init_meta(file_name_meta, n_nodes=n_nodes, n_epochs_max=n_epochs_max, train_ratio=train_ratio,
        batch_size=batch_size, n_workers=n_workers, index=index, log_scat=log_scat,
        n_filter_octave=n_filter_octave, avg_len=avg_len, device=device, samples=samples, root_dir=root_dir)

    labels = np.array(list(product(*labels)), dtype='float32').swapaxes(0, 1) # shaped (n_labels, n_conditions)
    labels = np.tile(labels[:, :, np.newaxis], [1, 1, n_samples_total]).reshape([n_labels, -1])
    for idx_label in range(n_labels):
        dataset = TimeSeriesDataset(data_scat, labels[idx_label], transform=ToTensor())
        dataloader = {phase:DataLoader(dataset, sampler=SubsetRandomSampler(index[phase]), batch_size=batch_size, num_workers=n_workers) for phase in phases}
        # train the neural network for the given idx_label
        _train(dataloader, n_data=n_data, n_nodes=n_nodes, device=device,
            n_epochs_max=n_epochs_max, file_name=file_name_meta, idx_label=idx_label,
            root_dir=root_dir)

def _init_meta(file_name, n_nodes, n_epochs_max, train_ratio, batch_size, n_workers, index,
    log_scat, n_filter_octave, avg_len, device, samples, root_dir=ROOT_DIR):
    '''initializes the dict type meta data prior to training the neural network'''

    file_name, _ = os.path.splitext(file_name)
    file_path = os.path.join(root_dir, file_name + '.pt')
    n_labels = len(samples['label_names'])

    meta = {'file_name_data':file_name + '.pt', 'root_dir':root_dir, 'n_nodes':n_nodes,
        'n_epochs_max':n_epochs_max, 'train_ratio':train_ratio, 'batch_size':batch_size,
        'n_workers':n_workers, 'index':index, 'log_scat':log_scat,
        'n_filter_octave':n_filter_octave, 'avg_len':avg_len, 'device':device,
        'loss_mean':[{'train':[], 'val':[]} for _ in range(n_labels)],
        'epoch':[[] for _ in range(n_labels)], 'weights':[[] for _ in range(n_labels)],
        'labels':samples['labels'], 'label_names':samples['label_names']}
    torch.save(meta, file_path)

def _train(dataloader, n_data, n_nodes, device, n_epochs_max, file_name, idx_label, root_dir=ROOT_DIR,
    lr=0.001, betas=(0.9, 0.999)):
    '''constructs and trains neural networks given the dataloader instance and network structure

    inputs
    ------
    dataloader - dict with keys train, val and values being dataloader instances
    n_data - dict with keys train, val and values being number of data samples
    n_nodes - list of number of nodes for the layers including the input and output layer   
    device - whether to run on gpu or cpu
    n_epochs_max - maximum number of epochs to run. can be terminated by KeyboardInterrupt
    file_name - name of file that contains the empty meta data
    idx_label - int representing which neural network to train
    root_dir - root directory of the meta data file

    outputs
    -------
    saves data into given file

    FIXME: perform scat transform outside this function so that this can be used in a more general sense
    '''
    file_name, _ = os.path.splitext(file_name)
    file_path = os.path.join(root_dir, file_name + '.pt')
    phases = ['train', 'val']
    assert(set(dataloader.keys()) == set(n_data.keys()) == set(phases)),\
        "Invalid keys given for either dataloader or n_data argument. Should be 'train' and 'val'"
    net = Net(n_nodes=n_nodes).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas)
    criterion = nn.MSELoss(reduction='sum')
    for epoch in range(n_epochs_max):
        try:
            loss_sum = {}
            loss_mean = {}
            for phase in phases:
                net.train(phase == 'train')
                loss_sum[phase] = 0.
                for batch in dataloader[phase]:
                    batch_data, batch_labels = batch['data'].to(device), batch['labels'].to(device)
                    output = net(batch_data)[:, 0] # output of net is shaped (batch_size, 1). drop dummy axis
                    loss = criterion(output, batch_labels)
                    optimizer.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    loss_sum[phase] += loss.data.item()
                loss_mean[phase] = loss_sum[phase] / n_data[phase] # MSE loss per data point
            if epoch % 10 == 0:
                loss_msg = ("{} out of {} epochs, mean_loss_train:{:.5f}, mean_loss_val:{:.5f}"
                    .format(epoch, n_epochs_max, loss_mean['train'], loss_mean['val']))
                print(loss_msg)
                meta = torch.load(file_path)
                meta['epoch'][idx_label].append(epoch)
                meta['weights'][idx_label].append(net.state_dict())
                for phase in phases:
                    meta['loss_mean'][idx_label][phase].append(loss_mean[phase])
                torch.save(meta, file_path)
        except KeyboardInterrupt:
            break

