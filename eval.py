import os
import argparse

from tqdm import tqdm
from tensorboard_logger import configure, log_value

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torch.utils.data
from torch.utils.data import DataLoader

from math import log10
import pandas as pd
import pytorch_ssim

from preprocess import DevDataset, display_transform
from model import Generator

def main():
	parser = argparse.ArgumentParser(description='Validate SRGAN')
	parser.add_argument('--dev_set', default='data/dev', type=str, help='dev set path')

	opt = parser.parse_args()
    dev_path = opt.dev_set

	dev_set = DevDataset(dev_path, upscale_factor=4)
	dev_loader = DataLoader(dataset=dev_set, num_workers=1, batch_size=1, shuffle=False)

    configure("tensorboard/srgan-val", flush_secs=5)
        
	netG = Generator()

	if torch.cuda.is_available():
		netG.cuda()
	
    for epoch in range(5, 101):
        if epoch % 5 == 0 :
            netG.load_state_dict(torch.load('epochs/netG_epoch_'+ str(epoch) +'_gpu.pth'))
        
            with torch.no_grad():
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
                    
                    # Only save 1 images to avoid out of memory
                    if len(dev_images) < 360 :
                        dev_images.extend([display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)), display_transform()(sr.data.cpu().squeeze(0))])
                
                dev_images = torch.stack(dev_images)
                dev_images = torch.chunk(dev_images, dev_images.size(0) // 9)
                
                dev_save_bar = tqdm(dev_images, desc='[saving training results]')
                index = 1
                for image in dev_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                    index += 1
                        
                log_value('ssim', cache['ssim']/len(dev_loader), epoch)
                log_value('psnr', cache['psnr']/len(dev_loader), epoch)
			
if __name__ == '__main__':
	main()
