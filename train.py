import os
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torch.utils.data
from torch.utils.data import DataLoader

import torchvision.utils as utils
from torchvision.models.vgg import vgg16

from preprocess import TrainDataset
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='SRGAN Train')
parser.add_argument('--crop_size', default=64, type=int, help='training images crop size')
parser.add_argument('--num_epochs', default=100, type=int, help='training epoch')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--train_set', default='data/train', type=str, help='train set path')

opt = parser.parse_args()

input_size = opt.crop_size
n_epoch = opt.num_epochs
batch_size = opt.batch_size

train_set = TrainDataset(opt.train_set, crop_size=input_size, upscale_factor=4)
train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=batch_size, shuffle=True)

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
	
if not torch.cuda.is_available():
	print ('!!!!!!!!!!!!!!USING CPU!!!!!!!!!!!!!')

netG = Generator(n_residual=4)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

for epoch in range(1, n_epoch + 1):
	train_bar = tqdm(train_loader)
	
	netG.train()
	netD.train()
    
	for data, target in train_bar:
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

		# Print information by tqdm
		train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, n_epoch, d_loss, g_loss))
            
	# Save model parameters
	if epoch == n_epoch or epoch%10 == 0:
		if torch.cuda.is_available():
			torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_gpu.pth' % (epoch))
			torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_gpu.pth' % (epoch))
		else:
			torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_cpu.pth' % (epoch))
			torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_cpu.pth' % (epoch))
