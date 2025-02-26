import argparse
import cv2
import numpy as np
import os

def increase_brightness(increment: int, input_path: str, output_path: str):
    """Increase brightness for all channels of the image"""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Input image not found at {input_path}")
    
    # Create a 3-channel increment array
    increment_array = np.array([increment, increment, increment], dtype=np.uint8)
    
    # Add increment to all channels
    adjusted = cv2.add(img, increment_array)
    cv2.imwrite(output_path, adjusted)

def decrease_brightness(decrement: int, input_path: str, output_path: str):
    """Decrease brightness for all channels of the image"""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Input image not found at {input_path}")
    
    # Create a 3-channel decrement array
    decrement_array = np.array([decrement, decrement, decrement], dtype=np.uint8)
    
    # Subtract decrement from all channels
    adjusted = cv2.subtract(img, decrement_array)
    cv2.imwrite(output_path, adjusted)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust image brightness')
    parser.add_argument('--action', required=True, choices=['increase', 'decrease'],
                       help='Choose to increase or decrease brightness')
    parser.add_argument('--value', type=int, required=True,
                       help='Positive integer value for brightness adjustment')
    parser.add_argument('--input', required=True,
                       help='Path to input image file')
    parser.add_argument('--output',
                       help='Path to output image file (default: input filename with "_out" suffix)')

    args = parser.parse_args()

    # Validate positive value
    if args.value < 0:
        parser.error("Value must be a positive integer")

    # Generate output path if not provided
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_out2_{args.action}{ext}"

    try:
        if args.action == 'increase':
            increase_brightness(args.value, args.input, output_path)
        else:
            decrease_brightness(args.value, args.input, output_path)
        print(f"Successfully processed image. Result saved to: {output_path}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")


# python .\brightness_adjustment2.py --action decrease --value 100 --input input.jpg --output dark2.jpg