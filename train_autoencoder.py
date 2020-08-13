import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os

import net_utils as nu

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

file_name = 'oocyte_stack_test.tif'
root_dir = './data/compression'
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) # to mean 0.5, std 0.5. should be [0.5] or (0.5,). If you just do 0.5, it'll interpret it differently
    # TODO: question: so this means different images will have a different linear mappings? (a,b in ax+b will be different since the images in batches will have different std values and means)
])

dataset = nu.ImageStackDataset(file_name=file_name, root_dir=root_dir, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = nu.Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
    weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img = data['data']
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
          .format(epoch+1, num_epochs, loss.data.item()))
    if epoch % 10 == 0:
        pic = nu.to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './dc_img/conv_autoencoder.pth')


