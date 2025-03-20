from gaussian_processing import *
from laplacian_pyramid import *

def reconstruct_gaussian_pyramid(laplacian_pyramid_path, gaussian_top_path, sigma=1.0, kernel_size=7):
    """Reconstruct Gaussian Pyramid from Laplacian Pyramid and the top Gaussian level."""
    # 1. Load Laplacian Pyramid (sorted by level)
    laplacian_pyramid = []

    for filename in os.listdir(laplacian_pyramid_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(laplacian_pyramid_path, filename)
            img = Image.open(img_path)
            img_array = np.array(img, dtype=np.float32) / 255.0
            laplacian_pyramid.append(img_array)
    
    # 2. Load the top Gaussian level
    top_gaussian = np.array(Image.open(gaussian_top_path), dtype=np.float32) / 255.0
    
    # 3. Reconstruct from top to bottom
    reconstructed_pyramid = [top_gaussian.copy()]
    for i in range(len(laplacian_pyramid)-1, 0, -1):  # build up-to-down
        current_gaussian = reconstructed_pyramid[len(reconstructed_pyramid) - 1]
        
        # Upsample the current Gaussian level
        upsampled = gaussian_upsampling(current_gaussian, sigma, kernel_size, scale_factor=2)
        
        # Adjust size to match Laplacian level (i-1)
        h, w = laplacian_pyramid[i-1].shape[:2]
        upsampled = upsampled[:h, :w] if upsampled.ndim == 2 else upsampled[:h, :w, :]
        
        # Add Laplacian level
        new_gaussian = laplacian_pyramid[i-1] + upsampled
        reconstructed_pyramid.append(new_gaussian)

    # 4. Save reconstructed Gaussian Pyramid
    output_dir = "reconstructed_gaussian_pyramid"
    os.makedirs(output_dir, exist_ok=True)
    # save in reverse
    for idx, img in enumerate(reconstructed_pyramid[::-1]):
        save_img = np.clip(img * 255, 0, 255).astype(np.uint8)
        filename = os.path.join(output_dir, f"recon_{idx+1}.png")
        Image.fromarray(save_img).save(filename)
        print(f"Saved: {filename}")
    
    print("Gaussian Pyramid Reconstructed!")

def main():
    parser = argparse.ArgumentParser(description='Reconstruct Gaussian Pyramid from Laplacian Pyramid.')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('-t', '--threshold', type=int, default=8,
                        help='Threshold size to stop downsampling (default: 8)')
    parser.add_argument('-s', '--sigma', type=float, default=1.0,
                        help='Sigma for Gaussian kernel (must match pyramid creation)')
    parser.add_argument('-k', '--kernel_size', type=int, default=7,
                        help='Kernel size (must match pyramid creation)')
    
    args = parser.parse_args()
    n_level = build_gaussian_pyramid(args.input, args.sigma, args.kernel_size, args.threshold)
    build_laplacian_pyramid("gaussian_pyramid", args.sigma, args.kernel_size)
    reconstruct_gaussian_pyramid("laplacian_pyramid", f"gaussian_pyramid/down_{n_level}.png", args.sigma, args.kernel_size)

if __name__ == "__main__":
    main()