import os
import numpy as np
import argparse
from PIL import Image

def create_gaussian_kernel(sigma, kernel_size):
    """create 2-dim gaussian blur kernel"""
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Initialize the kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # Calculate the values of the kernel
    sum_val = 0  # record the sum of kernel values for normalization
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            sum_val += kernel[i, j]

    # Normalize the kernel
    kernel /= np.sum(kernel)  

    return kernel

def convolve(image, kernel):
    """convolve image with kernel (zero padding)"""
    # We only process image with single or multiple channels,
    # which is 2-dim or 3-dim
    if len(image.shape) not in [2, 3]:
        raise ValueError("Image must be 2-dim or 3-dim")
    
    # If image is 3-dim, we compose it into several 2-dim images
    # and then convolve them separately by call itself recursively
    if len(image.shape) == 3:
        convolved_channels = []
        for c in range(image.shape[2]):
            convolved = convolve(image[:, :, c], kernel)
            convolved_channels.append(convolved)
        return np.stack(convolved_channels, axis=2)
    
    # ========== Single Channel Processing ==========
    kh, kw = kernel.shape
    ih, iw = image.shape

    # zero padding
    pad_h = kh // 2 
    pad_w = kw // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # convolution
    output = np.zeros((ih, iw))
    for i in range(ih):
        for j in range(iw):
            selected_region = padded_image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(selected_region * kernel)
    
    return output

def gaussian_downsampling(image, sigma=1.0, kernel_size=7):
    """2x gaussian downsampling (blur + downsample)"""
    # 1. create gaussian kernel
    kernel = create_gaussian_kernel(sigma, kernel_size)

    # 2. convolve image with kernel
    blurred_image = convolve(image, kernel)

    # 3. downsample the convolved image
    downsampled_image = blurred_image[::2, ::2]

    return downsampled_image


def gaussian_upsampling(image, sigma=1.0, kernel_size=7, scale_factor=2):
    """gaussian upsampling (upsample + blur)"""
    # 1. upsample the image
    h, w = image.shape[:2]
    new_h = h * scale_factor
    new_w = w * scale_factor

    if len(image.shape) == 2:
        upsampled_image = np.zeros((new_h, new_w), dtype=image.dtype)
        upsampled_image[::scale_factor, ::scale_factor] = image
    else:
        upsampled_image = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
        upsampled_image[::scale_factor, ::scale_factor, :] = image

    # 2. create gaussian kernel
    # TODO: check if the compensation is correct
    kernel = create_gaussian_kernel(sigma, kernel_size)
    kernel *= (scale_factor ** 2)  # compensate for the brightness loss due to zero insertion

    # 3. convolve the upsampled image with the kernel
    blurred_image = convolve(upsampled_image, kernel)

    return blurred_image

def build_gaussian_pyramid(image_path, sigma=1.0, kernel_size=7, threshold=8):
    # Read the image and convert it to float32
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32) / 255.0

    output_dir = "gaussian_pyramid"
    os.makedirs(output_dir, exist_ok=True)
    
    # Downsampling loop
    current = img_array.copy()
    down_steps = 0
    print("=== Downsampling Process ===")

    while True:
        h, w = current.shape[:2]
        print(f"Current size: {w}x{h}")

        if h <= threshold or w <= threshold:
            break

        current = gaussian_downsampling(current, sigma, kernel_size)
        down_steps += 1
        
        save_img = np.clip(current * 255, 0, 255).astype(np.uint8)
        filename = os.path.join(output_dir, f"down_{down_steps}.png")
        Image.fromarray(save_img).save(filename)
        print(f"Saved: {filename}")
    
    print("Gaussian Pyramid Built!")

    return down_steps

# TODO: Check the following examples
# 错误示例：先降采样再滤波
def wrong_downsample(image):
    downsampled = image[::2, ::2]  # 直接降采样
    kernel = create_gaussian_kernel(sigma=1.0, kernel_size=3)
    return convolve(downsampled, kernel)  # 后滤波

# 结果：混叠伪影明显（如边缘锯齿）

# 错误示例：先滤波再插值
def wrong_upsample(image):
    kernel = create_gaussian_kernel(sigma=1.0, kernel_size=3)
    blurred = convolve(image, kernel)
    h, w = blurred.shape[:2]
    upsampled = np.zeros((h*2, w*2))
    upsampled[::2, ::2] = blurred  # 插值
    return upsampled

# 结果：图像过度模糊且亮度不足

def main():
    parser = argparse.ArgumentParser(description='Gaussian Pyramid Processing Tool')
    parser.add_argument('--input', help='Input image path')
    parser.add_argument('-t', '--threshold', type=int, default=8,
                        help='Threshold size to stop downsampling (default: 8)')
    parser.add_argument('-s', '--sigma', type=float, default=1.0,
                        help='Sigma parameter for the Gaussian kernel (default: 1.0)')
    parser.add_argument('-k', '--kernel_size', type=int, default=7,
                        help='Gaussian kernel size (odd number, default: 7)')

    args = parser.parse_args()
    build_gaussian_pyramid(args.input, args.sigma, args.kernel_size, args.threshold)

if __name__ == "__main__":
    main()
