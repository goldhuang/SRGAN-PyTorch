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
        for param in self.perception_network.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        adversarial_loss = torch.mean(1 - out_labels)
        perception_loss = self.mse_loss(self.perception_network(out_images), self.perception_network(target_images))
        image_loss = self.mse_loss(out_images, target_images)
        
        return image_loss + 0.001 * adversarial_loss + 0.0001 * perception_loss

parser = argparse.ArgumentParser(description='SRGAN Train')
parser.add_argument('--crop_size', default=64, type=int, help='training images crop size')
parser.add_argument('--num_epochs', default=100, type=int, help='training epoch')

opt = parser.parse_args()

input_size = opt.crop_size
n_epoch = opt.num_epochs

train_set = TrainDataset('data/train', crop_size=input_size, upscale_factor=4)
train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=64, shuffle=True)

if torch.cuda.is_available() != True:
	print ('!!!!!!!!!!!!!!USING CUP!!!!!!!!!!!!!')

netG = Generator(8)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

lossG = GeneratorLoss()

if torch.cuda.is_available():
	netG.cuda()
	netD.cuda()
	lossG.cuda()

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

for epoch in range(1, n_epoch + 1):
	train_bar = tqdm(train_loader)
	cache = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
	
	netG.train()
	netD.train()
    
	for data, target in train_bar:
		batch_size = data.size(0)
		cache['batch_sizes'] += batch_size
        # Train D
		real_img = Variable(target)
		if torch.cuda.is_available():
			real_img = real_img.cuda()
		z = Variable(data)
		if torch.cuda.is_available():
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
		cache['g_loss'] += g_loss.item() * batch_size
		d_loss = 1 - real_out + fake_out
		cache['d_loss'] += d_loss.item() * batch_size
		cache['d_score'] += real_out.item() * batch_size
		cache['g_score'] += fake_out.item() * batch_size

		train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, n_epoch, cache['d_loss'] / cache['batch_sizes'],
            cache['g_loss'] / cache['batch_sizes'],
            cache['d_score'] / cache['batch_sizes'],
            cache['g_score'] / cache['batch_sizes']))
            
    # save model parameters
	torch.save(netG.state_dict(), 'epochs/netG_epoch_%d.pth' % (epoch))
	torch.save(netD.state_dict(), 'epochs/netD_epoch_%d.pth' % (epoch))
