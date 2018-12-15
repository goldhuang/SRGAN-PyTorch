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

from utils import DevDataset, to_image
from model import Generator

def main():
	parser = argparse.ArgumentParser(description='Validate SRGAN')
	parser.add_argument('--val_set', default='data/val', type=str, help='dev set path')
	parser.add_argument('--m0', default='cp/netG_SRGAN_gpu.pth', type=str, help='model0')
	parser.add_argument('--m1', default='cp/netG_SRWGANGP_gpu.pth', type=str, help='model1')
	
	opt = parser.parse_args()
	val_path = opt.val_set
	m0 = opt.m0
	m1 = opt.m1

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

	with torch.no_grad():
		netG.eval()

		val_bar = tqdm(val_loader)
		dev_images = []
		for val_lr, val_bic, val_hr in val_bar:
			batch_size = val_lr.size(0)

			if torch.cuda.is_available():
				lr = val_lr.cuda()
				hr = val_hr.cuda()
				
			netG.load_state_dict(torch.load(m0))	
			sr0 = netG(lr)
			
			netG.load_state_dict(torch.load(m1))	
			sr1 = netG(lr)
			
			netG.load_state_dict(torch.load('cp/netG_baseline_gpu.pth'))
			sr_baseline = netG(lr)
			
			# Avoid out of memory crash on 8G GPU
			if len(dev_images) < 80 :
				dev_images.extend([to_image()(val_bic.data.cpu().squeeze(0)), to_image()(sr_baseline.data.cpu().squeeze(0)), to_image()(sr0.data.cpu().squeeze(0)), to_image()(sr1.data.cpu().squeeze(0)), to_image()(hr.data.cpu().squeeze(0))])
	
		dev_images = torch.stack(dev_images)
		dev_images = torch.chunk(dev_images, dev_images.size(0) // 5)

		dev_save_bar = tqdm(dev_images, desc='[saving images]')
		index = 1
		for image in dev_save_bar:
			image = utils.make_grid(image, nrow=5, padding=5)
			utils.save_image(image, out_path + '%d.png' % (index), padding=5)
			index += 1
			
if __name__ == '__main__':
	main()
