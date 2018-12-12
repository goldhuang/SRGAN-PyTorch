# SRGAN for Anime 

A PyTorch implementation of SRGAN based on __Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network__ (https://arxiv.org/abs/1609.04802).
And another PyTorch WGAN-gp implementation of SRGAN referring to __Improved Training of Wasserstein GANs__ (https://arxiv.org/pdf/1704.00028.pdf).


## Requirements

* Python 3.*
* PyTorch
* torchvision
* tensorboard_logger
* tqdm
* CUDA* - only tested on Nvidia RTX 2070


## Datasets

11328 images from __kaggle dataset__ (https://www.kaggle.com/mylesoneill/tagged-anime-illustrations/home). Train/Dev/Test set sizes are 10816/256/256.

## Training

### Original SRGAN

```
python train.py --train_set=data/train
```
See more parameters in train.py.

### WGAN with gradient penalty

```
python train-wgangp.py --train_set=data/train
```
See more parameters in train-wgangp.py.


## Testing

```
python eval.py --val_set=data/val --start=1 --end=100 --interval=1
```
The sample command is to test with all the checkpoints from 1st to 100th epoch and print the results like the ones at the bottom of the page.
See more parameters in eval.py.


## Single Image Super Resulution

```
python sr.py --lr=lr.png
```
See more parameters in sr.py.

## Results

__Bicubic Upsampled / Original / GAN Upsampled / Deep ResNet Upsampled__

### WGAN-gp
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/WGAN-GP/1.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/WGAN-GP/2.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/WGAN-GP/3.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/WGAN-GP/4.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/WGAN-GP/5.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/WGAN-GP/6.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/WGAN-GP/7.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/WGAN-GP/8.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/WGAN-GP/9.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/WGAN-GP/10.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/WGAN-GP/11.png">

### SRGAN
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/SRGAN/1.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/SRGAN/2.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/SRGAN/3.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/SRGAN/4.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/SRGAN/5.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/SRGAN/6.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/SRGAN/7.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/SRGAN/8.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/SRGAN/9.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/SRGAN/10.png">
<img src="https://github.com/goldhuang/SRGAN-PyTorch/blob/master/results/SRGAN/11.png">
