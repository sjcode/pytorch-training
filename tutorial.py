import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5)) # transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

train_dataset = torchvision.datasets.FashionMNIST('./data',train=True,transform=transforms,download=True)
valid_dataset = torchvision.datasets.FashionMNIST('./data',train=False,transform=transforms,download=True)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=4, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=4, shuffle=False)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

print('Train datasets has {} instances'.format(len(train_dataset)))
print('Valid datasets has {} instances'.format(len(valid_dataset)))

import matplotlib.pyplot as plt
import numpy as np

def matplotlib_imshow(img, one_channle=False):
    if one_channle:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channle:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1,2,0))) # change 'hwc' order
    plt.show()
    plt.waitforbuttonpress(0)

dataiter = iter(train_loader)
images, labels = next(dataiter)

img_grid = torchvision.utils.make_grid(images)
#matplotlib_imshow(img_grid)
print(' '.join(classes[labels[j]] for j in range(4)))

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.nn.functional as F

class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = GarmentClassifier()
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_indx,tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print('    batch {} loss {}'.format(i+1, last_loss))
            tb_x = epoch_indx * len(train_loader) + i + 1
            # len(train_loader) it equals number of batches
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCH = 5
best_vloss = 1_000_000.

for epcoh in range(EPOCH):
    print('EPOCH {}'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    model.train(False)

    running_loss = 0.0
    for i, vdata in enumerate(valid_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)

        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_loss += vloss
    
    avg_vloss = running_loss / (i + 1)
    print('LOSS Train {} valid {}'.format(avg_loss, avg_vloss))

    writer.add_scalars('Training vs. Validation Loss',
                       {'Trainning': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.flush()

    if avg_vloss > best_vloss:
        best_vloss = avg_vloss
        import os
        model_path = os.path.join(os.path.dirname(__file__),'model_{}_{}'.format(timestamp, epoch_number))
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
