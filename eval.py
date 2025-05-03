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

# Add these functions at the top of your file, after imports:

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if not torch.cuda.is_available():
        return 0, 0
    
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)    # MB
    
    return allocated, reserved

def count_parameters(model):
    """Count number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_all_mb

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
	configure(str(now.tm_mon) + '-' + str(now.tm_mday) + '-' + str(now.tm_hour) + '-' + str(now.tm_min), flush_secs=5)
        
	netG = Generator()

	num_params = count_parameters(netG)
	model_size_mb = get_model_size(netG)
	print(f"\n===== Model Statistics =====")
	print(f"Parameters: {num_params:,}")
	print(f"Model Size: {model_size_mb:.2f} MB")
	print(f"===========================\n")

	if torch.cuda.is_available():
		netG.cuda()
		print(f"GPU: {torch.cuda.get_device_name(0)}")
		allocated, reserved = get_gpu_memory_usage()
		print(f"Initial GPU Memory: {allocated:.2f} MB (allocated), {reserved:.2f} MB (reserved)")
	
	out_path = 'vis/'
	if not os.path.exists(out_path):
		os.makedirs(out_path)
				
	for epoch in range(start, end+1):
		if epoch%interval == 0:
			with torch.no_grad():
				netG.eval()

				if torch.cuda.is_available():
					torch.cuda.reset_peak_memory_stats()

				val_bar = tqdm(val_loader)
				cache = {'ssim': 0, 'psnr': 0, 'inference_times': [], 'gpu_mem': []}
				dev_images = []
				for val_lr, val_hr_restore, val_hr in val_bar:
					batch_size = val_lr.size(0)

					lr = Variable(val_lr)
					hr = Variable(val_hr)
					hr_restore = Variable(val_hr_restore)
					if torch.cuda.is_available():
						lr = lr.cuda()
						hr = hr.cuda()
						hr_restore = hr_restore.cuda()
						
					netG.load_state_dict(torch.load('cp/netG_epoch_'+ str(epoch) +'_gpu.pth', weights_only=True))	
					
					start_time = time.time()
					sr = netG(lr)
					torch.cuda.synchronize()
					end_time = time.time()
					inference_time = end_time - start_time
					cache['inference_times'].append(inference_time)

					# Track GPU memory
					if torch.cuda.is_available():
						allocated, reserved = get_gpu_memory_usage()
						cache['gpu_mem'].append((allocated, reserved))

					psnr = 10 * log10(1 / ((sr - hr) ** 2).mean().item())
					ssim = pytorch_ssim.ssim(sr, hr).item()

					psnr_bicubic = 10 * log10(1 / ((hr_restore - hr) ** 2).mean().item())
					ssim_bicubic = pytorch_ssim.ssim(hr_restore, hr).item()

					val_bar.set_description(desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (psnr, ssim))
					val_bar.set_description(
        				desc='[SR] PSNR: %.4f dB SSIM: %.4f | [Bicubic] PSNR: %.4f dB SSIM: %.4f' 
        				% (psnr, ssim, psnr_bicubic, ssim_bicubic))
					
					cache['ssim'] += ssim
					cache['psnr'] += psnr
					
					netG.load_state_dict(torch.load('cp/netG_baseline_gpu.pth', weights_only=True))
					sr_baseline = netG(lr)
					
					# Avoid out of memory crash on 8G GPU
					if len(dev_images) < 80 :
						dev_images.extend([to_image()(val_hr_restore.squeeze(0)), to_image()(hr.data.cpu().squeeze(0)), to_image()(sr.data.cpu().squeeze(0)), to_image()(sr_baseline.data.cpu().squeeze(0))])


				avg_time = sum(cache['inference_times']) / len(cache['inference_times'])
				print(f"\nAverage inference time per image: {avg_time:.4f} seconds")
				print(f"FPS: {1.0/avg_time:.2f}")

				# Print GPU memory statistics
				if torch.cuda.is_available():
					peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
					peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

					avg_allocated = sum(m[0] for m in cache['gpu_mem']) / len(cache['gpu_mem'])
					avg_reserved = sum(m[1] for m in cache['gpu_mem']) / len(cache['gpu_mem'])

					print(f"GPU Memory (average): {avg_allocated:.2f} MB (allocated), {avg_reserved:.2f} MB (reserved)")
					print(f"GPU Memory (peak): {peak_allocated:.2f} MB (allocated), {peak_reserved:.2f} MB (reserved)")

				dev_images = torch.stack(dev_images)
				dev_images = torch.chunk(dev_images, dev_images.size(0) // 4)

				dev_save_bar = tqdm(dev_images, desc='[saving training results]')
				index = 1
				for image in dev_save_bar:
					image = utils.make_grid(image, nrow=4, padding=5)
					utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
					index += 1

				log_value('ssim', cache['ssim']/len(val_loader), epoch)
				log_value('psnr', cache['psnr']/len(val_loader), epoch)
			
if __name__ == '__main__':
	main()
