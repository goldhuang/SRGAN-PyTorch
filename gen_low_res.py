import os
from os import listdir, makedirs
from os.path import join, exists
from PIL import Image, ImageFilter
from io import BytesIO

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def generate_lowres_same_size(hr_dir, lr_dir, method='downscale_upscale', factor=4, jpeg_quality=15, blur_radius=2.0):
    """
    Generate low-resolution versions of images while preserving original dimensions
    
    Args:
        hr_dir: Directory with high-resolution images
        lr_dir: Directory to save low-resolution images
        method: How to degrade quality without changing size:
                'downscale_upscale' - downscale and upscale back (default)
                'jpeg_artifacts' - use JPEG compression artifacts
                'gaussian_blur' - apply Gaussian blur
        factor: Downscaling factor for 'downscale_upscale' method (default: 4)
        jpeg_quality: Quality setting for JPEG compression (1-100, lower=worse quality)
        blur_radius: Radius for Gaussian blur
    """
    if not exists(lr_dir):
        makedirs(lr_dir)
    
    hr_image_filenames = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
    
    print(f"Processing {len(hr_image_filenames)} images from {hr_dir} to {lr_dir}")
    print(f"Method: {method}")
    
    for i, hr_path in enumerate(hr_image_filenames):
        filename = os.path.basename(hr_path)
        
        # Load HR image
        hr_image = Image.open(hr_path).convert('RGB')
        original_size = hr_image.size
        
        if method == 'downscale_upscale':
            # Down-sample then up-sample to original size
            downscale_size = (original_size[0] // factor, original_size[1] // factor)
            
            # Ensure dimensions are at least 1 pixel
            if downscale_size[0] < 1: downscale_size = (1, downscale_size[1])
            if downscale_size[1] < 1: downscale_size = (downscale_size[0], 1)
            
            lr_image = hr_image.resize(downscale_size, Image.BICUBIC)
            lr_image = lr_image.resize(original_size, Image.BICUBIC)
            
        elif method == 'jpeg_artifacts':
            # Use JPEG compression to introduce artifacts
            buffer = BytesIO()
            hr_image.save(buffer, format="JPEG", quality=jpeg_quality)
            buffer.seek(0)
            lr_image = Image.open(buffer).convert('RGB')
            
        elif method == 'gaussian_blur':
            # Apply Gaussian blur to reduce detail
            lr_image = hr_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
        else:
            # Unknown method, just copy
            lr_image = hr_image.copy()
        
        # Save LR image with original dimensions
        lr_path = join(lr_dir, filename)
        lr_image.save(lr_path)
        
        # Print progress
        if (i + 1) % 10 == 0 or i == len(hr_image_filenames) - 1:
            print(f"Processed {i+1}/{len(hr_image_filenames)} images")
    
    print("Done!")
    
    # Display example images
    try:
        example_hr = Image.open(join(hr_dir, listdir(hr_dir)[0]))
        example_lr = Image.open(join(lr_dir, listdir(lr_dir)[0]))
        print(f"Example HR image size: {example_hr.size}")
        print(f"Example LR image size: {example_lr.size} (preserved dimensions)")
    except IndexError:
        print("No images found.")
    except Exception as e:
        print(f"Error checking example images: {e}")

if __name__ == "__main__":
    # Paths
    hr_dir = "data\\val"
    lr_dir = "data\\low_res"
    
    # Choose one method:
    # 'downscale_upscale' (recommended) - Most similar to true downsampling
    # 'jpeg_artifacts' - Creates compression artifacts
    # 'gaussian_blur' - Simply blurs the image
    
    method = 'downscale_upscale'
    
    # Additional parameters
    downscale_factor = 4       # How much to downscale before upscaling back
    jpeg_quality = 15          # 1-100, lower = worse quality
    blur_radius = 2.0          # Higher = more blurry
    
    # Generate low-res images with same dimensions
    generate_lowres_same_size(
        hr_dir, 
        lr_dir, 
        method=method,
        factor=downscale_factor,
        jpeg_quality=jpeg_quality,
        blur_radius=blur_radius
    )