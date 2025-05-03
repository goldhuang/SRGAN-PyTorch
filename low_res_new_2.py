import os
from os import listdir, makedirs
from os.path import join, exists
from PIL import Image
from tqdm import tqdm

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def generate_lr_images_preserve_size(hr_dir, lr_dir, upscale_factor=4):
    """
    Generate low-resolution images from high-resolution ones while preserving the original
    dimensions (just downscaled).
    
    Args:
        hr_dir: Directory containing high-resolution images
        lr_dir: Directory where low-resolution images will be saved
        upscale_factor: Downscaling factor
    """
    if not exists(lr_dir):
        makedirs(lr_dir)
    
    # Get all image files from hr_dir
    hr_image_filenames = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
    
    print(f"Processing {len(hr_image_filenames)} images from {hr_dir} to {lr_dir}")
    
    for i, hr_path in tqdm(enumerate(hr_image_filenames), total=len(hr_image_filenames), desc="Generating LR images"):
        try:
            # Get filename without path
            filename = os.path.basename(hr_path)
            
            # Load HR image
            hr_image = Image.open(hr_path).convert('RGB')
            
            # Get original size
            original_w, original_h = hr_image.size
            
            # Calculate new size
            new_w = original_w // upscale_factor
            new_h = original_h // upscale_factor
            
            # Create LR image by resizing with bicubic interpolation
            lr_image = hr_image.resize((new_w, new_h), Image.BICUBIC)
            
            # Save LR image
            lr_path = join(lr_dir, filename)
            lr_image.save(lr_path)
            
        except Exception as e:
            print(f"Error processing {hr_path}: {e}")
    
    print("Done!")
    
    # Display example image dimensions
    if hr_image_filenames:
        example_hr = Image.open(hr_image_filenames[0])
        example_lr = Image.open(join(lr_dir, os.path.basename(hr_image_filenames[0])))
        print(f"Example HR image size: {example_hr.size}")
        print(f"Example LR image size: {example_lr.size}")
        print(f"Downscale factor: {example_hr.width / example_lr.width:.2f}x")

if __name__ == "__main__":
    # Directories
    hr_dir = "data/val"
    lr_dir = "data/lr_val_full"
    
    # Parameters
    upscale_factor = 4  # Same as in the SRGAN model
    
    # Generate low-res images
    generate_lr_images_preserve_size(hr_dir, lr_dir, upscale_factor)