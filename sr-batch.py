import argparse
import time
import os

from os.path import basename, normpath

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='SR single image')
parser.add_argument('--lr', type=str, help='test image path')
parser.add_argument('--start', default=1, type=int, help='model start')
parser.add_argument('--end', default=100, type=int, help='model end')
parser.add_argument('--interval', default=1, type=int, help='model end')
parser.add_argument('--output', default='batch', type=str, help='sub folder')
opt = parser.parse_args()

lr = opt.lr
start = opt.start
end = opt.end
interval = opt.interval

with torch.no_grad():
	model = Generator().eval()
	if torch.cuda.is_available():
		model.cuda()
	
	for epoch in range(start, end+1):
		if epoch%interval == 0:
			model.load_state_dict(torch.load('cp/netG_epoch_'+ str(epoch) +'_gpu.pth'))

			image = Image.open(lr)

			image = Variable(ToTensor()(image)).unsqueeze(0)
	
			if torch.cuda.is_available():
				image = image.cuda()

			out = model(image)
			out_img = ToPILImage()(out[0].data.cpu())
			out_path = 'generated/' + opt.output + '/'
			if not os.path.exists(out_path):
				os.makedirs(out_path)
			out_img.save(out_path + str(epoch) +basename(normpath(lr)))