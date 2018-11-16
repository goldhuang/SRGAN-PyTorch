import os
import argparse

import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Generator, Discriminator

import torch
import torch.nn as nn
import torchvision.models.vgg.vgg16 as vgg16

from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def hr_preprocess(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def lr_preprocess(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])
    
class TrainDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_preprocess = hr_preprocess(crop_size)
        self.lr_preprocess = lr_preprocess(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_preprocess(Image.open(self.image_filenames[index]))
        lr_image = self.lr_preprocess(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        self.perception_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        adversarial_loss = torch.mean(1 - out_labels)
        perception_loss = self.mse_loss(self.perception_network(out_images), self.perception_network(target_images))
        image_loss = self.mse_loss(out_images, target_images)
        
        return image_loss + 0.001 * adversarial_loss + 0.001 * perception_loss

parser = argparse.ArgumentParser(description='SRGAN Train')
parser.add_argument('--crop_size', default=64, type=int, help='training images crop size')
parser.add_argument('--num_epochs', default=100, type=int, help='training epoch')

opt = parser.parse_args()

input_size = opt.crop_size
n_epoch = opt.num_epochs

train_set = TrainDataset('data/train', crop_size=input_size, upscale_factor=4)
train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=64, shuffle=True)

netG = Generator(8)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

lossG = GeneratorLoss()

netG.cuda()
netD.cuda()
lossG.cuda()

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

for epoch in range(1, n_epoch + 1):
    train_bar = tqdm(train_loader)

    netG.train()
    netD.train()
    
    for data, target in train_bar:
    
        # Train D
        real_img = Variable(target)
        real_img = real_img.cuda()
        z = Variable(data)
        z = z.cuda()
        fake_img = netG(z)

        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        # Train G
        netG.zero_grad()
        g_loss = lossG(fake_out, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()
        fake_img = netG(z)
        fake_out = netD(fake_img).mean()

        g_loss = lossG(fake_out, fake_img, real_img)
        d_loss = 1 - real_out + fake_out

    # save model parameters
    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
