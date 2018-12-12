import os
import argparse
import time
import numpy as np

from tqdm import tqdm
from tensorboard_logger import configure, log_value

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data
from torch.utils.data import DataLoader

import torchvision.utils as utils
from torchvision.transforms import Normalize

from math import log10
import pytorch_ssim

from model import Generator, Discriminator

from utils import TrainDataset, DevDataset, to_image, print_first_parameter, check_grads, get_grads_D, get_grads_G

def main():
	n_epoch_pretrain = 2
	use_tensorboard = True

	parser = argparse.ArgumentParser(description='SRGAN Train')
	parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
	parser.add_argument('--num_epochs', default=1000, type=int, help='training epoch')
	parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
	parser.add_argument('--train_set', default='data/train', type=str, help='train set path')
	parser.add_argument('--check_point', type=int, default=-1, help="continue with previous check_point")

	opt = parser.parse_args()

	input_size = opt.crop_size
	n_epoch = opt.num_epochs
	batch_size = opt.batch_size
	check_point = opt.check_point

	check_point_path = 'cp/'
	if not os.path.exists(check_point_path):
		os.makedirs(check_point_path)

	train_set = TrainDataset(opt.train_set, crop_size=input_size, upscale_factor=4)
	train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=batch_size, shuffle=True)

	dev_set = DevDataset('data/dev', upscale_factor=4)
	dev_loader = DataLoader(dataset=dev_set, num_workers=1, batch_size=1, shuffle=False)

	mse = nn.MSELoss()
	bce = nn.BCELoss()
	#tv = TVLoss()
		
	if not torch.cuda.is_available():
		print ('!!!!!!!!!!!!!!USING CPU!!!!!!!!!!!!!')

	netG = Generator()
	print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
	netD = Discriminator()
	print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

	if torch.cuda.is_available():
		netG.cuda()
		netD.cuda()
		#tv.cuda()
		mse.cuda()
		bce.cuda()
	
	if use_tensorboard:
		configure('log', flush_secs=5)
	
	# Pre-train generator using only MSE loss
	if check_point == -1:
		optimizerG = optim.Adam(netG.parameters())
		#schedulerG = MultiStepLR(optimizerG, milestones=[20], gamma=0.1)
		for epoch in range(1, n_epoch_pretrain + 1):
			#schedulerG.step()		
			train_bar = tqdm(train_loader)
			
			netG.train()
			
			cache = {'g_loss': 0}
			
			for lowres, real_img_hr in train_bar:
				if torch.cuda.is_available():
					real_img_hr = real_img_hr.cuda()
					
				if torch.cuda.is_available():
					lowres = lowres.cuda()
					
				fake_img_hr = netG(lowres)

				# Train G
				netG.zero_grad()
				
				image_loss = mse(fake_img_hr, real_img_hr)
				cache['g_loss'] += image_loss
				
				image_loss.backward()
				optimizerG.step()

				# Print information by tqdm
				train_bar.set_description(desc='[%d/%d] Loss_G: %.4f' % (epoch, n_epoch_pretrain, image_loss))
				
		# Save model parameters	
		#if torch.cuda.is_available():
		#	torch.save(netG.state_dict(), 'cp/netG_epoch_pre_gpu.pth')
		#else:
		#	torch.save(netG.state_dict(), 'cp/netG_epoch_pre_cpu.pth')
	
	optimizerG = optim.Adam(netG.parameters())
	optimizerD = optim.Adam(netD.parameters())
	
	if check_point != -1:
		if torch.cuda.is_available():
			netG.load_state_dict(torch.load('cp/netG_epoch_' + str(check_point) + '_gpu.pth'))
			netD.load_state_dict(torch.load('cp/netD_epoch_' + str(check_point) + '_gpu.pth'))
			optimizerG.load_state_dict(torch.load('cp/optimizerG_epoch_' + str(check_point) + '_gpu.pth'))
			optimizerD.load_state_dict(torch.load('cp/optimizerD_epoch_' + str(check_point) + '_gpu.pth'))
		else :
			netG.load_state_dict(torch.load('cp/netG_epoch_' + str(check_point) + '_cpu.pth'))
			netD.load_state_dict(torch.load('cp/netD_epoch_' + str(check_point) + '_cpu.pth'))
			optimizerG.load_state_dict(torch.load('cp/optimizerG_epoch_' + str(check_point) + '_cpu.pth'))
			optimizerD.load_state_dict(torch.load('cp/optimizerD_epoch_' + str(check_point) + '_cpu.pth'))
	
	for epoch in range(1 + max(check_point, 0), n_epoch + 1 + max(check_point, 0)):
		train_bar = tqdm(train_loader)
		
		netG.train()
		netD.train()
		
		cache = {'mse_loss': 0, 'tv_loss': 0, 'adv_loss': 0, 'g_loss': 0, 'd_loss': 0, 'ssim': 0, 'psnr': 0, 'd_top_grad' : 0, 'd_bot_grad' : 0, 'g_top_grad' : 0, 'g_bot_grad' : 0}
		
		for lowres, real_img_hr in train_bar:
			#print ('lr size : ' + str(data.size()))
			#print ('hr size : ' + str(target.size()))
			if torch.cuda.is_available():
				real_img_hr = real_img_hr.cuda()
				lowres = lowres.cuda()
			
			# Train D
			
			#if not check_grads(netD, 'D'):
			#	return
			netD.zero_grad()
			
			logits_real = netD(real_img_hr)
			logits_fake = netD(netG(lowres).detach())
			
			# Lable smoothing
			real = torch.tensor(torch.rand(logits_real.size())*0.25 + 0.85)
			fake = torch.tensor(torch.rand(logits_fake.size())*0.15)
			
			# Lable flipping
			prob = (torch.rand(logits_real.size()) < 0.05)
			
			#print ('logits real size : ' + str(logits_real.size()))
			#print ('logits fake size : ' + str(logits_fake.size()))
			
			if torch.cuda.is_available():
				real = real.cuda()
				fake = fake.cuda()
				prob = prob.cuda()
				
			real_clone = real.clone()
			real[prob] = fake[prob]
			fake[prob] = real_clone[prob]
            
			d_loss = bce(logits_real, real) + bce(logits_fake, fake)
			
			cache['d_loss'] += d_loss.item()
			
			d_loss.backward()
			optimizerD.step()
			
			dtg, dbg = get_grads_D(netD)

			cache['d_top_grad'] += dtg
			cache['d_bot_grad'] += dbg

			# Train G
					
			#if not check_grads(netG, 'G'):
			#	return
			netG.zero_grad()
			
			fake_img_hr = netG(lowres)
			image_loss = mse(fake_img_hr, real_img_hr)
			
			logits_fake_new = netD(fake_img_hr)
			adversarial_loss = bce(logits_fake_new, torch.ones_like(logits_fake_new))
			
			#tv_loss = tv(fake_img_hr)
			
			g_loss = image_loss + 1e-2*adversarial_loss

			cache['mse_loss'] += image_loss.item()
			#cache['tv_loss'] += tv_loss.item()
			cache['adv_loss'] += adversarial_loss.item()
			cache['g_loss'] += g_loss.item()

			g_loss.backward()
			optimizerG.step()
			
			gtg, gbg = get_grads_G(netG)

			cache['g_top_grad'] += gtg
			cache['g_bot_grad'] += gbg

			# Print information by tqdm
			train_bar.set_description(desc='[%d/%d] D grads:(%f, %f) G grads:(%f, %f) Loss_D: %.4f Loss_G: %.4f = %.4f + %.4f' % (epoch, n_epoch, dtg, dbg, gtg, gbg, d_loss, g_loss, image_loss, adversarial_loss))
		
		if use_tensorboard:
			log_value('d_loss', cache['d_loss']/len(train_loader), epoch)
		
			log_value('mse_loss', cache['mse_loss']/len(train_loader), epoch)
			#log_value('tv_loss', cache['tv_loss']/len(train_loader), epoch)
			log_value('adv_loss', cache['adv_loss']/len(train_loader), epoch)
			log_value('g_loss', cache['g_loss']/len(train_loader), epoch)
			
			log_value('D top layer gradient', cache['d_top_grad']/len(train_loader), epoch)
			log_value('D bot layer gradient', cache['d_bot_grad']/len(train_loader), epoch)
			log_value('G top layer gradient', cache['g_top_grad']/len(train_loader), epoch)
			log_value('G bot layer gradient', cache['g_bot_grad']/len(train_loader), epoch)
		
		# Save model parameters	
		if torch.cuda.is_available():
			torch.save(netG.state_dict(), 'cp/netG_epoch_%d_gpu.pth' % (epoch))
			if epoch%5 == 0:
				torch.save(netD.state_dict(), 'cp/netD_epoch_%d_gpu.pth' % (epoch))
				torch.save(optimizerG.state_dict(), 'cp/optimizerG_epoch_%d_gpu.pth' % (epoch))
				torch.save(optimizerD.state_dict(), 'cp/optimizerD_epoch_%d_gpu.pth' % (epoch))
		else:
			torch.save(netG.state_dict(), 'cp/netG_epoch_%d_cpu.pth' % (epoch))
			if epoch%5 == 0:
				torch.save(netD.state_dict(), 'cp/netD_epoch_%d_cpu.pth' % (epoch))
				torch.save(optimizerG.state_dict(), 'cp/optimizerG_epoch_%d_cpu.pth' % (epoch))
				torch.save(optimizerD.state_dict(), 'cp/optimizerD_epoch_%d_cpu.pth' % (epoch))
				
		# Visualize results
		with torch.no_grad():
			netG.eval()
			out_path = 'vis/'
			if not os.path.exists(out_path):
				os.makedirs(out_path)
				
			dev_bar = tqdm(dev_loader)
			valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
			dev_images = []
			for val_lr, val_hr_restore, val_hr in dev_bar:
				batch_size = val_lr.size(0)
				lr = val_lr
				hr = val_hr
				if torch.cuda.is_available():
					lr = lr.cuda()
					hr = hr.cuda()
				
				sr = netG(lr)
				
				psnr = 10 * log10(1 / ((sr - hr) ** 2).mean().item())
				ssim = pytorch_ssim.ssim(sr, hr).item()
				dev_bar.set_description(desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (psnr, ssim))
				
				cache['ssim'] += ssim
				cache['psnr'] += psnr
				
				# Avoid out of memory crash on 8G GPU
				if len(dev_images) < 60 :
					dev_images.extend([to_image()(val_hr_restore.squeeze(0)), to_image()(hr.data.cpu().squeeze(0)), to_image()(sr.data.cpu().squeeze(0))])
			
			dev_images = torch.stack(dev_images)
			dev_images = torch.chunk(dev_images, dev_images.size(0) // 3)
			
			dev_save_bar = tqdm(dev_images, desc='[saving training results]')
			index = 1
			for image in dev_save_bar:
				image = utils.make_grid(image, nrow=3, padding=5)
				utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
				index += 1
		
			if use_tensorboard:			
				log_value('ssim', cache['ssim']/len(dev_loader), epoch)
				log_value('psnr', cache['psnr']/len(dev_loader), epoch)
			
if __name__ == '__main__':
	main()
