import os
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import scatnet as scn
import scatnet_utils as scu

ROOT_DIR = './data/'

class Net(nn.Module):
    def __init__(self, n_nodes):
        super(Net, self).__init__()
        net = []
        for idx in range(len(n_nodes) - 2):
            net.append(nn.Linear(n_nodes[idx], n_nodes[idx + 1]))
            net.append(nn.ELU(inplace=True))
            net.append(nn.BatchNorm1d(n_nodes[idx + 1]))
            net.append(nn.Dropout())
        net.append(nn.Linear(n_nodes[-2], n_nodes[-1]))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class LSTM(nn.Module):
    def __init__(self, params):
        super(LSTM, self).__init__()
        pass


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


def train(file_name, n_nodes_hidden, n_epochs_max, train_ratio, avg_len, log_scat=True, n_filter_octave=[1, 1], batch_size=100, root_dir=ROOT_DIR, n_workers=4):
    file_path = os.path.join(root_dir, file_name)
    samples = torch.load(file_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    file_name, _ = os.path.splitext(file_name)
    file_path_meta = os.path.join(root_dir, file_name + '.pt')

    data, labels, label_names = samples['data'], samples['labels'], samples['label_names']
    n_data_train_val, n_channels, data_len = data.shape[-3:]
    n_labels = len(label_names) # number of labels to predict
    phases = ['train', 'val']
    n_data = {'train':int(n_data_train_val * train_ratio)}
    n_data['val'] = n_data_train_val - n_data['train']
    idx_train = np.random.choice(n_data_train_val, n_data['train'], replace=False).tolist()
    idx_val = list(set(range(n_data_train_val)) - set(idx_train))
    data = {'train':np.take(data, idx_train, axis=-3), 'val':np.take(data, idx_val, axis=-3)}

    # flatten the data
    data = {phase:np.reshape(data[phase], [-1, n_channels, data_len]) for phase in phases}
    # perform scattering transform
    data_scat = {}
    for phase in phases:
        scat = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
        S = scat.transform(data[phase])
        if log_scat: S = scu.log_scat(S)
        S = scu.stack_scat(S)
        S = S.mean(axis=-1)
        S = np.reshape(S, (-1, S.shape[-1]))
        data_scat[phase] = S

    n_scat_nodes = data_scat['train'].shape[-1]
    n_nodes = [n_scat_nodes] + list(n_nodes_hidden) + [1]
    meta = {'file_name_data':file_name, 'root_dir':root_dir, 'n_nodes_hidden':n_nodes_hidden, 'n_epochs_max':n_epochs_max,
        'train_ratio':train_ratio, 'batch_size':batch_size, 'n_workers':n_workers,
        'loss_mean':{'train':[], 'val':[]}, 'epoch':[], 'weights':[]}

    labels = np.array(list(product(*labels)), dtype='float32').swapaxes(0, 1)
    labels = {phase:np.repeat(np.expand_dims(labels, axis=-1).reshape([n_labels, -1]), n_data[phase], axis=-1) for phase in phases}
    for idx in range(n_labels):
        dataset = {phase:TimeSeriesDataset(data_scat[phase], labels[phase][idx], transform=ToTensor()) for phase in phases}
        dataloader = {phase:DataLoader(dataset[phase], batch_size=batch_size, shuffle=True, num_workers=n_workers) for phase in phases}

        net = Net(n_nodes=n_nodes).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.MSELoss(reduction='sum')

        for epoch in range(n_epochs_max):
            loss_sum = {}
            loss_mean = {}
            for phase in phases:
                net.train(phase=='train')
                loss_sum[phase] = 0.
                for batch in dataloader[phase]:
                    batch_data, batch_labels = batch['data'].to(device), batch['labels'].to(device)
                    output = net(batch_data)
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
                meta['epoch'].append(epoch)
                meta['weights'].append(net.state_dict())
                for phase in phases:
                    meta['loss_mean'][phase].append(loss_mean[phase])
                torch.save(meta, file_path_meta)
