import os
from itertools import product
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import common_utils as cu
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
            # nn.BatchNorm1d() can only be processed for batch size being larger than 1.
            # setting to be non-functional for now.
            #net.append(nn.BatchNorm1d(n_nodes[idx + 1]))
            #net.append(nn.Dropout())
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


def train(file_name, n_nodes_hidden, avg_len, log_scat=True, n_filter_octave=[1, 1], n_epochs_max=2000, train_ratio=0.8, batch_size=100, n_workers=4, root_dir=ROOT_DIR):
    file_name, _ = os.path.splitext(file_name)
    file_path = os.path.join(root_dir, file_name + '.pt')
    samples = torch.load(file_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nums = cu.match_filename(r'{}_meta_nn_([0-9]+).pt'.format(file_name), root_dir=root_dir)
    nums = [int(num) for num in nums]
    idx = max(nums) + 1 if nums else 0
    file_path_meta = os.path.join(root_dir, '{}_meta_nn_{}.pt'.format(file_name, idx))

    data, labels, label_names = samples['data'], samples['labels'], samples['label_names']
    n_conditions = np.prod(data.shape[:-3])
    n_samples_train_val, n_channels, data_len = data.shape[-3:]
    n_labels = len(label_names) # number of labels to predict
    phases = ['train', 'val']
    n_samples = {'train':int(n_samples_train_val * train_ratio)}
    n_samples['val'] = n_samples_train_val - n_samples['train']
    n_samples['total'] = n_samples_train_val
    idx_train = np.random.choice(n_samples_train_val, n_samples['train'], replace=False)
    idx_val = np.array(list(set(range(n_samples_train_val)) - set(idx_train)))
    index = {'train':idx_train, 'val':idx_val}
    n_data = {phase:n_samples[phase] * n_conditions for phase in phases}
    n_data['total'] = n_data['train'] + n_data['val']

    # flatten the data to shape (n_data['total'], n_channels, data_len)
    data = np.reshape(data, [-1, n_channels, data_len])
    # perform scattering transform
    scat = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
    S = scat.transform(data)
    if log_scat: S = scu.log_scat(S)
    S = scu.stack_scat(S) # shaped (n_data['total'], n_channels, n_scat_nodes, data_len)
    data_scat = np.reshape(S, (n_data['total'], -1)) # shaped (n_data['total'], n_channels * n_scat_nodes * data_len)

    n_nodes_input_layer = data_scat.shape[-1]
    n_nodes = [n_nodes_input_layer] + list(n_nodes_hidden) + [1]
    meta = {'file_name_data':file_name + '.pt', 'root_dir':root_dir, 'n_nodes':n_nodes,
        'n_epochs_max':n_epochs_max, 'train_ratio':train_ratio, 'batch_size':batch_size,
        'n_workers':n_workers, 'index':index, 'log_scat':log_scat,
        'n_filter_octave':n_filter_octave, 'avg_len':avg_len, 'device':device,
        'loss_mean':[{'train':[], 'val':[]} for _ in range(n_labels)],
        'epoch':[[] for _ in range(n_labels)], 'weights':[[] for _ in range(n_labels)],
        'labels':samples['labels'], 'label_names':samples['label_names']}
    # NOTE: labels and label_names can also be fetched from the data file

    torch.save(meta, file_path_meta)
    labels = np.array(list(product(*labels)), dtype='float32').swapaxes(0, 1) # shaped (n_labels, n_conditions)
    labels = np.tile(labels[:, :, np.newaxis], [1, 1, n_samples['total']]).reshape([n_labels, -1])
    for idx_label in range(n_labels):
        dataset = TimeSeriesDataset(data_scat, labels[idx_label], transform=ToTensor())
        dataloader = {phase:DataLoader(dataset, sampler=SubsetRandomSampler(index[phase]), batch_size=batch_size, num_workers=n_workers) for phase in phases}

        net = Net(n_nodes=n_nodes).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
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
                    meta = torch.load(file_path_meta)
                    meta['epoch'][idx_label].append(epoch)
                    meta['weights'][idx_label].append(net.state_dict())
                    for phase in phases:
                        meta['loss_mean'][idx_label][phase].append(loss_mean[phase])
                    torch.save(meta, file_path_meta)
            except KeyboardInterrupt:
                break
                
