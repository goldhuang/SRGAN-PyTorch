import os
import argparse
import time

from tqdm import tqdm
from tensorboard_logger import configure, log_value

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision.utils as utils

import torch.utils.data
from torch.utils.data import DataLoader

from math import log10
import pandas as pd
import pytorch_ssim

from utils import DevDataset, to_image
from model import Generator

def main():
	parser = argparse.ArgumentParser(description='Validate SRGAN')
	parser.add_argument('--val_set', default='data/val', type=str, help='dev set path')
	parser.add_argument('--start', default=1, type=int, help='model start')
	parser.add_argument('--end', default=100, type=int, help='model end')
	parser.add_argument('--interval', default=1, type=int, help='model end')
	
	opt = parser.parse_args()
	val_path = opt.val_set
	start = opt.start
	end = opt.end
	interval = opt.interval

	val_set = DevDataset(val_path, upscale_factor=4)
	val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)
	
	now = time.gmtime(time.time())
	#configure(str(now.tm_mon) + '-' + str(now.tm_mday) + '-' + str(now.tm_hour) + '-' + str(now.tm_min), flush_secs=5)
        
	netG = Generator()

	if torch.cuda.is_available():
		netG.cuda()
	
	out_path = 'vis/'
	if not os.path.exists(out_path):
		os.makedirs(out_path)
				
	for epoch in range(start, end+1):
		if epoch%interval == 0:
			with torch.no_grad():
				netG.eval()

				val_bar = tqdm(val_loader)
				cache = {'ssim': 0, 'psnr': 0}
				dev_images = []
				for val_lr, val_hr_restore, val_hr in val_bar:
					batch_size = val_lr.size(0)

					lr = Variable(val_lr)
					hr = Variable(val_hr)
					if torch.cuda.is_available():
						lr = lr.cuda()
						hr = hr.cuda()
						
					netG.load_state_dict(torch.load('cp/netG_epoch_'+ str(epoch) +'_gpu.pth'))	
					sr = netG(lr)

					#psnr = 10 * log10(1 / ((sr - hr) ** 2).mean().item())
					#ssim = pytorch_ssim.ssim(sr, hr).item()
					#val_bar.set_description(desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (psnr, ssim))
				
					#cache['ssim'] += ssim
					#cache['psnr'] += psnr
					
					netG.load_state_dict(torch.load('cp/netG_baseline_gpu.pth'))
					sr_baseline = netG(lr)
					
					# Avoid out of memory crash on 8G GPU
					if len(dev_images) < 80 :
						dev_images.extend([to_image()(val_hr_restore.squeeze(0)), to_image()(hr.data.cpu().squeeze(0)), to_image()(sr.data.cpu().squeeze(0)), to_image()(sr_baseline.data.cpu().squeeze(0))])
			
				dev_images = torch.stack(dev_images)
				dev_images = torch.chunk(dev_images, dev_images.size(0) // 4)

				dev_save_bar = tqdm(dev_images, desc='[saving training results]')
				index = 1
				for image in dev_save_bar:
					image = utils.make_grid(image, nrow=4, padding=5)
					utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
					index += 1

				#log_value('ssim', cache['ssim']/len(val_loader), epoch)
				#log_value('psnr', cache['psnr']/len(val_loader), epoch)
			
if __name__ == '__main__':
	main()
