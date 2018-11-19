import argparse
import time

from os.path import basename, normpath

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='SR single image')
parser.add_argument('--lr_image', type=str, help='test image path')
parser.add_argument('--model_start', default=80, type=int, help='model start')
parser.add_argument('--model_end', default=300, type=int, help='model end')
opt = parser.parse_args()

lr = opt.lr_image
start = opt.model_start
end = opt.model_end

with torch.no_grad():
	model = Generator().eval()
	if torch.cuda.is_available():
		model.cuda()
	
	for epoch in range(start, end+1):
		if epoch%5 == 0:
			model.load_state_dict(torch.load('epochs/netG_epoch_'+ str(epoch) +'_gpu.pth'))

			image = Image.open(lr)

			image = Variable(ToTensor()(image)).unsqueeze(0)
	
			if torch.cuda.is_available():
				image = image.cuda()

			out = model(image)
			out_img = ToPILImage()(out[0].data.cpu())
			out_img.save('generated/' + str(epoch) +basename(normpath(lr)))