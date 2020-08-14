import torch
import torchvision
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image
import os
from skimage.external import tifffile
import net_utils as nu
import common_utils as cu
import time
import numpy as np

#file_name = 'oocyte_test_32px.tif'
file_name = 'density.tif'
root_dir = './data/compression'

n_epochs_max = 2000
batch_size = 512
train_ratio = 0.8
learning_rate = 1e-3
n_workers = 4
save_fig = True
idx_test_img = 3
save_model = True
lr = 0.001
betas = (0.9, 0.999)

# set data file name
file_name_noext, _ = os.path.splitext(file_name)
file_path = os.path.join(root_dir, file_name)
imgs = tifffile.imread(file_path)
max_val = imgs.max().astype('float32')
test_img = imgs[idx_test_img][np.newaxis, np.newaxis, :, :].astype('float32')
test_img = test_img / max_val

test_img = (test_img - 0.5) / 0.5
test_img = torch.tensor(test_img, dtype=torch.get_default_dtype())
test_imgs = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set meta file name
nums = cu.match_filename(r'{}_meta_autoconv_([0-9]+).pt'.format(file_name_noext), root_dir=root_dir)
nums = [int(num) for num in nums]; idx = max(nums) + 1 if nums else 0
file_name_meta = '{}_meta_autoconv_{}.pt'.format(file_name_noext, idx)
file_path_meta = os.path.join(root_dir, file_name_meta)

# define dataset
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # to mean 0.5, std 0.5. should be [0.5] or (0.5,). If you just do 0.5, it'll interpret it differently
    # TODO: question: so this means different images will have a different linear mappings? (a,b in ax+b will be different since the images in batches will have different std values and means)
])

dataset = nu.ImageStackDataset(file_name=file_name, root_dir=root_dir, transform=img_transform)
n_px = np.prod(list(dataset[0:1]['data'].shape[-2:]))
# split dataset into training and validation and get indices
n_data_total = len(dataset)
index = nu._train_test_split(n_data_total, train_ratio); index['val'] = index.pop('test')
n_data = {phase:len(index[phase]) for phase in ['train', 'val']}

# get dataloader
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader = {phase:DataLoader(dataset, sampler=SubsetRandomSampler(index[phase]),
    batch_size=batch_size, num_workers=n_workers) for phase in ['train', 'val']}

# initialize meta data and save it to a file
meta = {'file_name':file_name_meta, 'root_dir':root_dir,
    'n_epochs_max':n_epochs_max, 'train_ratio':train_ratio, 'batch_size':batch_size,
    'n_workers':n_workers, 'index':index, 'device':device, 
    }

meta.update({'epoch':[], 'model':[],
    'elapsed':[], 'loss':{'train':[], 'val':[]}, 'criterion':'rmse'})
nu._init_meta(**meta)

# define model, optimizer, criterion
model = nu.Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
criterion = nn.MSELoss(reduction='sum')

# start training
time_start = time.time()
metric = 'RMSE'
for epoch in range(n_epochs_max):
    loss_sum = {}
    loss_metric = {}
    for phase in ['train', 'val']:
        model.train(phase == 'train')
        loss_sum[phase] = 0.
        for batch in dataloader[phase]:
            batch_data = batch['data'].to(device)
            output = model(batch_data)
            # output = output[:, 0] # TODO: check size for autoencoder
            loss = criterion(output, batch_data)
            optimizer.zero_grad()
            if phase == 'train':
                loss.backward()
                optimizer.step()
            loss_sum[phase] += loss.data.item()
        loss_metric[phase] = np.sqrt(loss_sum[phase] / n_data[phase] / n_px)
    if epoch % 50 == 0:
        time_curr = time.time()
        elapsed = time_curr - time_start
        loss_msg = ("\t{} out of {} epochs, {}_train:{:.15f}, {}_val:{:.15f}, elapsed seconds:{:.2f}"
            .format(epoch, n_epochs_max, metric, loss_metric['train'], metric, loss_metric['val'], elapsed))
        print(loss_msg)
        meta = torch.load(file_path_meta)

        meta['epoch'].append(epoch)
        meta['elapsed'].append(elapsed)
        if save_model: meta['model'].append(model.state_dict())
        for phase in ['train', 'val']:
            meta['loss'][phase].append(loss_metric[phase])
        torch.save(meta, file_path_meta)
        if save_fig:
            output_test = nu.to_img(model(torch.tensor(test_img).to(device)).cpu()).detach().numpy() * max_val
            output_test = output_test.astype('uint8')
            test_imgs.append(output_test)
            test_stack = np.stack(test_imgs, axis=0)
            tifffile.imsave(os.path.join(root_dir, 'test_stack.tif'), test_stack)

