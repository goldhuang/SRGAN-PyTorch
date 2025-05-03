import os
from os import listdir, makedirs
from os.path import join, exists
from PIL import Image, ImageFilter
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def generate_blurred_images(hr_dir, blur_dir, blur_type='gaussian', strength=2.0, resize_factor=1.0):
    """
    Generate blurred versions of images from hr_dir and save to blur_dir
    
    Args:
        hr_dir: Directory with high-resolution images
        blur_dir: Directory to save blurred images
        blur_type: Type of blur to apply ('gaussian', 'box', 'motion', 'mixed')
        strength: Blur strength/radius
        resize_factor: Optional resize factor (<1.0 to make images smaller)
    """
    if not exists(blur_dir):
        makedirs(blur_dir)
    
    hr_image_filenames = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
    
    print(f"Processing {len(hr_image_filenames)} images from {hr_dir} to {blur_dir}")
    print(f"Blur type: {blur_type}, Strength: {strength}")
    
    for i, hr_path in enumerate(hr_image_filenames):
        filename = os.path.basename(hr_path)
        
        # Load HR image
        hr_image = Image.open(hr_path).convert('RGB')
        original_size = hr_image.size
        
        # Apply optional resizing
        if resize_factor != 1.0:
            new_size = (int(original_size[0] * resize_factor), int(original_size[1] * resize_factor))
            hr_image = hr_image.resize(new_size, Image.BICUBIC)
        
        # Apply blur based on selected type
        if blur_type == 'gaussian':
            # Standard Gaussian blur
            blurred_image = hr_image.filter(ImageFilter.GaussianBlur(radius=strength))
            
        elif blur_type == 'box':
            # Box blur (simpler, more uniform)
            box_size = int(strength * 2)
            blurred_image = hr_image.filter(ImageFilter.BoxBlur(box_size))
            
        elif blur_type == 'motion':
            # Motion blur (simulates camera movement)
            # For motion blur we use multiple directional box blurs
            kernel_size = int(strength * 3)
            kernel_size = max(kernel_size, 3)  # Ensure at least size 3
            
            # Simple horizontal motion blur using PIL
            blurred_image = hr_image.filter(
                ImageFilter.Kernel(
                    (kernel_size, kernel_size), 
                    [1/kernel_size if i == kernel_size//2 else 0 for j in range(kernel_size) for i in range(kernel_size)],
                    1
                )
            )
            
        elif blur_type == 'mixed':
            # Apply multiple blur types for more realistic degradation
            # First downsize and upsize to reduce details
            temp_size = (original_size[0] // 4, original_size[1] // 4)
            temp_img = hr_image.resize(temp_size, Image.BICUBIC)
            temp_img = temp_img.resize(hr_image.size, Image.BICUBIC)
            
            # Then apply Gaussian blur
            blurred_image = temp_img.filter(ImageFilter.GaussianBlur(radius=strength))
            
        else:
            # Unknown method, just copy
            blurred_image = hr_image.copy()
        
        # Save blurred image
        blur_path = join(blur_dir, filename)
        blurred_image.save(blur_path)
        
        # Print progress
        if (i + 1) % 10 == 0 or i == len(hr_image_filenames) - 1:
            print(f"Processed {i+1}/{len(hr_image_filenames)} images")
    
    print("Done!")
    
    # Display example images
    try:
        example_hr = Image.open(join(hr_dir, listdir(hr_dir)[0]))
        example_blur = Image.open(join(blur_dir, listdir(blur_dir)[0]))
        print(f"Example HR image size: {example_hr.size}")
        print(f"Example blurred image size: {example_blur.size}")
    except IndexError:
        print("No images found.")
    except Exception as e:
        print(f"Error checking example images: {e}")

if __name__ == "__main__":
    # Paths
    hr_dir = "data\\val"
    blur_dir = "data\\low_res_blur"
    
    # Choose one blur type:
    # 'gaussian' - Standard Gaussian blur
    # 'box' - Box blur (more uniform)
    # 'motion' - Motion blur (simulates camera movement)
    # 'mixed' - Combination of downscaling and blurring
    blur_type = 'mixed'
    
    # Parameters
    blur_strength = 2.5       # Blur radius/strength
    resize_factor = 1.0       # Keep original size
    
    # Generate blurred images
    generate_blurred_images(
        hr_dir, 
        blur_dir, 
        blur_type=blur_type,
        strength=blur_strength,
        resize_factor=resize_factor
    )