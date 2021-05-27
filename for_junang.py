import os
from itertools import product
import numpy as np

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image
from skimage.external import tifffile



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


class autoencoder(nn.Module):
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


def to_img(x): # this is for after the training
    x = 0.5 * (x + 1) # shifts things towards to the right a bit. num range: (0,1) -> (0,1) but useful in the case where the numbers are slightly outside this range (since you train it it doesn't exactly become (0,1))
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x







if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)) # to mean 0.5, std 0.5
    # TODO: question: so this means different images will have a different linear mappings? (a,b in ax+b will be different since the images in batches will have different std values and means)
])

#dataset = MNIST('./data', transform=img_transform)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
    weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')





