# usage: python laplacian_pyramid.py --input gaussian_pyramid -t 8 -s 1.0 -k 7

from gaussian_processing import *

def build_laplacian_pyramid(gaussian_pyramid_path, sigma=1.0, kernel_size=7):
    """build laplacian pyramid"""
    # 1. Load gaussian pyramid
    gaussian_pyramid = []

    for filename in os.listdir(gaussian_pyramid_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(gaussian_pyramid_path, filename)
            img = Image.open(img_path)
            img_array = np.array(img, dtype=np.float32) / 255.0
            gaussian_pyramid.append(img_array)

    # 2. Build laplacian pyramid
    laplacian_pyramid = []

    # build laplacian pyramid except the highest level
    for i in range(len(gaussian_pyramid) - 1):
        g_current = gaussian_pyramid[i]
        g_next = gaussian_pyramid[i + 1]

        # upsample the higher level image
        g_next_up = gaussian_upsampling(g_next, sigma, kernel_size, 2)

        # Ensure the shapes match (for odd sized situations)
        h, w = g_current.shape[:2]
        g_next_up = g_next_up[:h, :w] if g_next_up.ndim == 2 else g_next_up[:h, :w, :]

        # compute laplacian
        laplacian = g_current - g_next_up

        laplacian_pyramid.append(laplacian)

    # the highest level is the same as the original image
    laplacian_pyramid.append(gaussian_pyramid[-1].copy())

    print("Laplacian Pyramid Built!")

    # 3. save normalized laplacian pyramid
    output_dir = "laplacian_pyramid"
    os.makedirs(output_dir, exist_ok=True)

    offset = 0.5  # assume laplacian is in [-0.5, 0.5]
    for i, image in enumerate(laplacian_pyramid):
        save_img = np.clip((image + offset) * 255, 0, 255).astype(np.uint8)
        filename = os.path.join(output_dir, f"laplacian_{i}.png")
        Image.fromarray(save_img).save(filename)

def main():
    parser = argparse.ArgumentParser(description='Gaussian Pyramid Processing Tool')
    parser.add_argument('--input', type=str, default="gaussian_pyramid", 
                        help='Input image path(dafault: gaussian_pyramid)')
    parser.add_argument('-t', '--threshold', type=int, default=8,
                        help='Threshold size to stop downsampling (default: 8)')
    parser.add_argument('-s', '--sigma', type=float, default=1.0,
                        help='Sigma parameter for the Gaussian kernel (default: 1.0)')
    parser.add_argument('-k', '--kernel_size', type=int, default=7,
                        help='Gaussian kernel size (odd number, default: 7)')

    args = parser.parse_args()
    build_laplacian_pyramid(args.input, args.sigma, args.kernel_size)

if __name__ == "__main__":
    main()