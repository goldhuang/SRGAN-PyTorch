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
parser.add_argument('--model_name', default='epochs/netG_epoch_100_gpu.pth', type=str, help='model')
opt = parser.parse_args()

lr = opt.lr_image
pth = opt.model_name

model = Generator().eval()
if torch.cuda.is_available():
	model.cuda()
model.load_state_dict(torch.load(pth))

image = Image.open(lr)
with torch.no_grad():
	image = Variable(ToTensor()(image)).unsqueeze(0)
	
if torch.cuda.is_available():
    image = image.cuda()

out = model(image)
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('generated/' + basename(normpath(lr)))