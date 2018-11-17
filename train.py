import os
from os import listdir
from os.path import join
import argparse

from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torchvision.utils as utils
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torchvision.models.vgg import vgg16

from model import Generator, Discriminator

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def hr_preprocess(crop_size):
    return Compose([
    	# CenterCrop(256)
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

parser = argparse.ArgumentParser(description='SRGAN Train')
parser.add_argument('--crop_size', default=64, type=int, help='training images crop size')
parser.add_argument('--num_epochs', default=100, type=int, help='training epoch')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')

opt = parser.parse_args()

input_size = opt.crop_size
n_epoch = opt.num_epochs
batch_size = opt.batch_size

train_set = TrainDataset('data/train', crop_size=input_size, upscale_factor=4)
train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=batch_size, shuffle=True)

if torch.cuda.is_available() != True:
	print ('!!!!!!!!!!!!!!USING CUP!!!!!!!!!!!!!')

netG = Generator(n_residual=4)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

mse = nn.MSELoss()
bce = nn.BCELoss()

vgg = vgg16(pretrained=True)
netV = nn.Sequential(*list(vgg.features)[:31]).eval()
for param in netV.parameters():
    param.requires_grad = False

if torch.cuda.is_available():
	netG.cuda()
	netD.cuda()
	netV.cuda()
	mse.cuda()
	bce.cuda()

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

for epoch in range(1, n_epoch + 1):
	train_bar = tqdm(train_loader)
	
	netG.train()
	netD.train()
    
	for data, target in train_bar:
		cache = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
		
		batch_size = data.size(0)
		cache['batch_sizes'] += batch_size
		
		real_img_hr = Variable(target)
		if torch.cuda.is_available():
			real_img_hr = real_img_hr.cuda()
			
		lowres = Variable(data)
		if torch.cuda.is_available():
			lowres = lowres.cuda()
		fake_img_hr = netG(lowres)
		
		logits_real = netD(real_img_hr)
		logits_fake = netD(fake_img_hr)
			
        # Train D
		netD.zero_grad()
		
		d_loss = bce(logits_real, torch.ones_like(logits_real)) + bce(logits_fake, torch.zeros_like(logits_fake))
		
		d_loss.backward(retain_graph=True)
		optimizerD.step()

        # Train G
		netG.zero_grad()
		
		image_loss = mse(fake_img_hr, real_img_hr)
		perception_loss = mse(netV(fake_img_hr), netV(real_img_hr))
		adversarial_loss = bce(logits_fake, torch.ones_like(logits_fake))
		g_loss = image_loss + 0.4*2e-6*perception_loss + 1e-3*adversarial_loss

		g_loss.backward()
		optimizerG.step()

		train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, n_epoch, d_loss, g_loss))
            
	# save model parameters
	if epoch == n_epoch or epoch%10 == 0:
		torch.save(netG.state_dict(), 'epochs/netG_epoch_%d.pth' % (epoch))
		torch.save(netD.state_dict(), 'epochs/netD_epoch_%d.pth' % (epoch))
