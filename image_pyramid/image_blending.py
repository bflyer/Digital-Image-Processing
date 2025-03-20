# usage:
# python .\image_blending.py\
#  --image1 .\input_images\apple.jpg\
#  --image2 .\input_images\orange.jpg\
#  --mask .\input_images\mask.png

from gaussian_processing import *
from laplacian_pyramid import *
from image_reconstruction import *

def blend_laplacians(laplacian1, laplacian2, gaussian_mask, save=False):
    """
        blend two laplacian pyramids based on mask weights
        all input images have been devided by 255
    """
    
    blended = []
    for L1, L2, G in zip(laplacian1, laplacian2, gaussian_mask):
        if G.ndim == 2 and L1.ndim == 3:
            G = G[..., np.newaxis]
        blended.append(L1 * G + L2 * (1 - G))

    if save:
        output_dir = "blend_laplacian_pyramid"
        os.makedirs(output_dir, exist_ok=True)

        offset = 0.5  # assume laplacian is in [-0.5, 0.5]
        for i, image in enumerate(blended):
            save_img = np.clip((image + offset) * 255, 0, 255).astype(np.uint8)
            filename = os.path.join(output_dir, f"laplacian_{i}.png")
            Image.fromarray(save_img).save(filename)

    return blended

def main():
    parser = argparse.ArgumentParser(description='Blend images using Laplacian pyramids')
    parser.add_argument('--image1', type=str, required=True, help='Path to first image')
    parser.add_argument('--image2', type=str, required=True, help='Path to second image')
    parser.add_argument('--mask', type=str, required=True, help='Path to mask image')
    parser.add_argument('-t', '--threshold', type=int, default=8, help='Size threshold for downsampling')
    parser.add_argument('-s', '--sigma', type=float, default=1.0, help='Sigma for Gaussian kernel')
    parser.add_argument('-k', '--kernel_size', type=int, default=7, help='Kernel size for Gaussian')
    
    args = parser.parse_args()

    # create gaussian pyramid
    gp1 = build_gaussian_pyramid(args.image1, args.sigma, args.kernel_size, args.threshold)
    gp2 = build_gaussian_pyramid(args.image2, args.sigma, args.kernel_size, args.threshold)
    gp_mask = build_gaussian_pyramid(args.mask, args.sigma, args.kernel_size, args.threshold, gray_style=True)

    # create laplacian pyramid
    lp1 = build_laplacian_pyramid(gp1, args.sigma, args.kernel_size)
    lp2 = build_laplacian_pyramid(gp2, args.sigma, args.kernel_size)

    # blend laplacian pyramids
    blended_lp = blend_laplacians(lp1, lp2, gp_mask, save=True)

    # reconstruction
    reconstruct_gaussian_pyramid(blended_lp, args.sigma, args.kernel_size)

    print("Blended image reconstructed!")

if __name__ == "__main__":
    main()