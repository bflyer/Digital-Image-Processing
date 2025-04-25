import argparse
from PIL import Image
import sys

from utils import *

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Seam Carving for Content-Aware Image Resizing')
    parser.add_argument('input', type=str, help='Input image file path')
    parser.add_argument('output', type=str, help='Output image file path')
    
    group = parser.add_mutually_exclusive_group()
    
    # mode1：target width and height
    group.add_argument('--size', metavar=('WIDTH', 'HEIGHT'), 
                      nargs=2, type=int,
                      help='Target size as width height (e.g., --size 800 600)')
    
    # mode2：delta width and height
    group.add_argument('--delta', metavar=('DW', 'DH'), 
                      nargs=2, type=int,
                      help='Size change as delta_width delta_height (e.g., --delta -100 50)')
    
    args = parser.parse_args()
    
    if args.size is None and args.delta is None:
        parser.error('You must specify either --size or --delta')
    
    # Load image
    try:
        img = Image.open(args.input)
    except IOError:
        print(f"Error: Cannot open image file {args.input}")
        sys.exit(1)
    
    h, w = img.size
    print(f"Image shape: ({h}, {w})")
    # Determine if we need to add or remove seams
    if args.size is not None:
        target_width, target_height = args.size
        delta_width = target_width - w
        delta_height = target_height - h
    else:
        delta_width, delta_height = args.delta
    
    # Perform seam carving
    result = seam_carve(img, delta_width, delta_height)
    
    # Save result
    result.save(args.output)
    print(f"Image successfully resized and saved to {args.output}")


if __name__ == "__main__":
    main()
    
# python seam_carving/image_resizing.py input input/rider.png output output/rider.png --size 800 600
# python seam_carving/image_resizing.py --input input/rider.png --output output/rider.png --delta -100 0
# python seam_carving/image_resizing.py seam_carving\input\rider.png seam_carving\output\rider.png --delta -100 0
# python seam_carving/image_resizing.py seam_carving\output\rider.png seam_carving\output\rider
# 
# 
# python seam_carving/image_resizing.py seam_carving\output\rider.png seam_carving\output\rider2.png --delta 100 0