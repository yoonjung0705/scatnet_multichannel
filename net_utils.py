import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, file_name, root_dir=ROOT_DIR):
        self._file_name = file_name
        self._root_dir = root_dir
        self._data = cu.load_data(file_name, root_dir=root_dir)
        self._len = len(self._data)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.data.iloc[index, :]

    def fit(self, file_name):
        pass

    def transform(self, file_name):
        pass

    def fit_transform(self, file_name):
        self.fit(file_name)
        self.transform(file_name)


class ToTensor:
    def __call__(self, data):
        return torch.tensor(data)



def train(file_name, root_dir=ROOT_DIR, 
    file_path = os.path.join(root_dir, file_name)
    torch.load(file_path)



