import os
import argparse
import time

from tqdm import tqdm
from tensorboard_logger import configure, log_value

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torch.utils.data
from torch.utils.data import DataLoader

import torchvision.utils as utils
from torchvision.models.vgg import vgg16

from math import log10
import pandas as pd
import pytorch_ssim

from preprocess import TrainDataset, DevDataset, display_transform
from model import Generator, Discriminator

def main():
	n_epoch_pretrain = 100
	use_tensorboard = True

	parser = argparse.ArgumentParser(description='SRGAN Train')
	parser.add_argument('--crop_size', default=64, type=int, help='training images crop size')
	parser.add_argument('--num_epochs', default=100, type=int, help='training epoch')
	parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
	parser.add_argument('--train_set', default='data/train', type=str, help='train set path')
	parser.add_argument('--check_point', type=str, default='', help="check point name to load")

	opt = parser.parse_args()

	input_size = opt.crop_size
	n_epoch = opt.num_epochs
	batch_size = opt.batch_size
	check_point = opt.check_point

	train_set = TrainDataset(opt.train_set, crop_size=input_size, upscale_factor=4)
	train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=batch_size, shuffle=True)

	dev_set = DevDataset('data/dev', upscale_factor=4)
	dev_loader = DataLoader(dataset=dev_set, num_workers=1, batch_size=1, shuffle=False)

	mse = nn.MSELoss()
	bce = nn.BCELoss()

	vgg = vgg16(pretrained=True)
	netV = nn.Sequential(*list(vgg.features)[:31]).eval()
	for param in netV.parameters():
		param.requires_grad = False
		
	if not torch.cuda.is_available():
		print ('!!!!!!!!!!!!!!USING CPU!!!!!!!!!!!!!')

	netG = Generator()
	print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
	netD = Discriminator()
	print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

	if check_point != '':
		netG.load_state_dict(torch.load('epochs/netG_' + check_point + '.pth'))
		
		d_path = 'epochs/netD_' + check_point + '.pth'
		if os.path.isfile(d_path): 
			netD.load_state_dict(torch.load())

	if torch.cuda.is_available():
		netG.cuda()
		netD.cuda()
		netV.cuda()
		mse.cuda()
		bce.cuda()

	optimizerG = optim.Adam(netG.parameters())
	optimizerD = optim.Adam(netD.parameters())
	
	if use_tensorboard:
		configure("tensorboard/srgan-" + str(n_epoch_pretrain) + '-' + str(n_epoch), flush_secs=5)
	
	start_time = time.process_time()
	
	# Pre-train generator using only MSE loss
	if check_point == '':
		for epoch in range(1, n_epoch_pretrain + 1):
			train_bar = tqdm(train_loader)
			
			netG.train()
			
			cache = {'g_loss': 0}
			
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

				# Train G
				netG.zero_grad()
				
				image_loss = mse(fake_img_hr, real_img_hr)
				cache['g_loss'] += image_loss
				
				image_loss.backward()
				optimizerG.step()

				# Print information by tqdm
				train_bar.set_description(desc='[%d/%d] Loss_G: %.4f' % (epoch, n_epoch_pretrain, image_loss))
			
			if use_tensorboard:
				log_value('pretrain-gloss', cache['g_loss']/len(train_loader), epoch)
	
	
	# Save model parameters	
	if torch.cuda.is_available():
		torch.save(netG.state_dict(), 'epochs/netG_epoch_pre_gpu.pth')
	else:
		torch.save(netG.state_dict(), 'epochs/netG_epoch_pre_cpu.pth')
		
	pretrain_done_time = time.process_time()	
	pretrain_time = pretrain_done_time - start_time
		
	for epoch in range(1, n_epoch + 1):
		train_bar = tqdm(train_loader)
		
		netG.train()
		netD.train()
		
		cache = {'image_loss': 0, 'perception_loss': 0, 'adversarial_loss': 0, 'g_loss': 0, 'd_loss_t': 0, 'd_loss_f': 0, 'd_loss': 0, 'ssim': 0, 'psnr': 0}
		
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
			
			d_loss_t = bce(logits_real, torch.ones_like(logits_real))
			d_loss_f = bce(logits_fake, torch.zeros_like(logits_fake))
			d_loss = d_loss_t + d_loss_f
			
			cache['d_loss_t'] += d_loss_t.item()
			cache['d_loss_f'] += d_loss_f.item()
			cache['d_loss'] += d_loss.item()
			
			d_loss.backward(retain_graph=True)
			optimizerD.step()

			# Train G
			netG.zero_grad()
			
			image_loss = mse(fake_img_hr, real_img_hr)
			perception_loss = mse(netV(fake_img_hr), netV(real_img_hr))
			adversarial_loss = bce(logits_fake, torch.ones_like(logits_fake))
			g_loss = image_loss + 1e-4*perception_loss + 1e-3*adversarial_loss

			cache['image_loss'] += image_loss.item()
			cache['perception_loss'] += perception_loss.item()
			cache['adversarial_loss'] += adversarial_loss.item()
			cache['g_loss'] += g_loss.item()

			g_loss.backward()
			optimizerG.step()

			# Print information by tqdm
			train_bar.set_description(desc='[%d/%d] Loss_D: %.4f = %.4f + %.4f Loss_G: %.4f = %.4f + %.4f + %.4f' % (epoch, n_epoch, d_loss, d_loss_t, d_loss_f, g_loss, image_loss, perception_loss, adversarial_loss))
		
		if use_tensorboard:
			log_value('d_loss_t', cache['d_loss_t']/len(train_loader), epoch)
			log_value('d_loss_f', cache['d_loss_f']/len(train_loader), epoch)
			log_value('d_loss', cache['d_loss']/len(train_loader), epoch)
		
			log_value('image_loss', cache['image_loss']/len(train_loader), epoch)
			log_value('perception_loss', cache['perception_loss']/len(train_loader), epoch)
			log_value('adversarial_loss', cache['adversarial_loss']/len(train_loader), epoch)
			log_value('g_loss', cache['g_loss']/len(train_loader), epoch)
		
		if epoch == n_epoch or epoch%5 == 0:
			# Save model parameters	
			if torch.cuda.is_available():
				torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_gpu.pth' % (epoch))
				torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_gpu.pth' % (epoch))
			else:
				torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_cpu.pth' % (epoch))
				torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_cpu.pth' % (epoch))
				
			# Visualize results
			if True:
				netG.eval()
				out_path = 'visualizaton/'
				if not os.path.exists(out_path):
					os.makedirs(out_path)
					
				dev_bar = tqdm(dev_loader)
				valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
				dev_images = []
				for val_lr, val_hr_restore, val_hr in dev_bar:
					batch_size = val_lr.size(0)
					valing_results['batch_sizes'] += batch_size
					with torch.no_grad():
						lr = Variable(val_lr)
						hr = Variable(val_hr)
					if torch.cuda.is_available():
						lr = lr.cuda()
						hr = hr.cuda()
					sr = netG(lr)

					batch_mse = ((sr - hr) ** 2).data.mean().item()
					valing_results['mse'] += batch_mse * batch_size
					batch_ssim = pytorch_ssim.ssim(sr, hr).item()
					valing_results['ssims'] += batch_ssim * batch_size
					valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
					valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
					dev_bar.set_description(
						desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
							valing_results['psnr'], valing_results['ssim']))
					
					cache['ssim'] += valing_results['ssim']
					cache['psnr'] += valing_results['psnr']
					
					# Only save 5 images to avoid out of memory 
					if len(dev_images) < 5 :
						dev_images.extend([display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)), display_transform()(sr.data.cpu().squeeze(0))])
				
				dev_images = torch.stack(dev_images)
				dev_images = torch.chunk(dev_images, dev_images.size(0) // 15)
				
				
				dev_save_bar = tqdm(dev_images, desc='[saving training results]')
				index = 1
				for image in dev_save_bar:
					image = utils.make_grid(image, nrow=3, padding=5)
					utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
					index += 1
			
			if use_tensorboard:			
				log_value('ssim', cache['ssim']/len(dev_loader), epoch)
				log_value('psnr', cache['psnr']/len(dev_loader), epoch)			
	
	train_done_time = time.process_time()	
	train_time = train_done_time - pretrain_done_time

	print ('pretrain time : %d s, train time : %d s' % (pretrain_time, train_time))
			
if __name__ == '__main__':
	main()
