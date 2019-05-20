import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb

class Net(nn.Module):
    def __init__(self, n_features, n_hidden_1, n_hidden_2, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden_1)
        self.fc1_dropout = nn.Dropout()
        self.fc1_bn = nn.BatchNorm1d(n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc2_dropout = nn.Dropout()
        self.fc2_bn = nn.BatchNorm1d(n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_output)

    def forward(self, x):
        # x = x.view(-1, self.num_flat_features(x))
        x = F.relu( self.fc1_bn( self.fc1_dropout( self.fc1(x))))
        x = F.relu( self.fc2_bn( self.fc2_dropout( self.fc2(x))))
        x = self.fc3(x)

        return x

    def num_flat_features(self,x ):
        size = x.size()[1:]
        num_features = 1
        for s in size:
           num_features *= s
        return num_features

