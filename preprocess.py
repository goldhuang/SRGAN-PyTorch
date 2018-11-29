import os
from os import listdir
from os.path import join

from PIL import Image

import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torchvision.utils as utils
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def to_image():
    return Compose([
        ToPILImage(),
        ToTensor()
    ])
    
def hr_preprocess_train(crop_size):
    return Compose([
    	CenterCrop(384),
        RandomCrop(crop_size),
        ToTensor(),
        Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    ])

def lr_preprocess_train(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor(),
        Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    ])
	
class TrainDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_preprocess = hr_preprocess_train(crop_size)
        self.lr_preprocess = lr_preprocess_train(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_preprocess(Image.open(self.image_filenames[index]))
        lr_image = self.lr_preprocess(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

def hr_preprocess_test(crop_size):
    return Compose([
    	Resize(crop_size, interpolation=Image.BICUBIC)
        ToTensor(),
        Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    ])

def lr_preprocess_test(crop_size, upscale_factor):
    return Compose([
        Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        ToTensor(),
        Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    ])
        
class TestDataset(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(DevDataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
		
        crop_size = calculate_valid_crop_size(128, self.upscale_factor)
        lr_scale = lr_preprocess_test(crop_size, self.upscale_factor)
        hr_scale = hr_preprocess_test(crop_size)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return lr_image, hr_restore_img, hr_image

    def __len__(self):
        return len(self.image_filenames)
		
#val_loader = torch.utils.data.DataLoader(
#    datasets.ImageFolder(valdir, transforms.Compose([
#        transforms.TenCrop(224),
#        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
#    ])),
#    batch_size=args.batch_size, shuffle=False,
#    num_workers=args.workers, pin_memory=True)
