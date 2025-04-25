import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from utils import *

from mask_generation import InteractiveMaskGenerator

def create_mask_from_user(image_path, mask_path=None):
    """
    Create or load a mask for the object to be removed.
    If mask_path is provided, load it. Otherwise, use interactive drawing.
    
    :param image_path: Path to input image
    :param mask_path: Optional path to existing mask
    :return: Binary mask (1=remove, 0=keep) as numpy array
    """
    if mask_path:
        try: 
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask) > 0
            return mask
        except FileNotFoundError:
            print("Mask file not found.")
    
    # Interactive mask creation
    print("Launching interactive mask creation...")
    mask_generator = InteractiveMaskGenerator(image_path, mask_path)
    mask = mask_generator.run()
    
    if mask is None:
        print("Mask creation cancelled by user.")
        return None
    
    return mask > 0  # Convert to boolean

def find_seam_to_remove_mask(energy, mask):
    """
    Find the seam that removes the most mask pixels.
    
    :param energy: Energy map (H, W)
    :param mask: Binary mask (H, W) where 1=remove, 0=keep
    :return: (seam, mask_pixels_removed), where seam is vertical or horizontal
    """
    # Find vertical seam that removes most mask pixels
    h, w = energy.shape
    vertical_seam = find_vertical_seam(energy)
    vertical_mask_removed = sum(mask[i, vertical_seam[i]] for i in range(h))
    
    # Find horizontal seam that removes most mask pixels
    horizontal_seam = find_horizontal_seam(energy)
    horizontal_mask_removed = sum(mask[horizontal_seam[j], j] for j in range(w))
    
    # Choose the seam that removes more mask pixels
    if vertical_mask_removed >= horizontal_mask_removed:
        return (vertical_seam, 'vertical'), vertical_mask_removed
    else:
        return (horizontal_seam, 'horizontal'), horizontal_mask_removed

def remove_object_with_mask(image, mask):
    """
    Remove the masked object by repeatedly removing seams that cover the most mask pixels.
    
    :param image: Input image (numpy array)
    :param mask: Binary mask (H, W) where 1=remove, 0=keep
    :return: Image with object removed, and number of seams removed in each direction
    """
    img = image.copy()
    mask = mask.copy()
    
    # Count how many seams we remove in each direction
    vertical_seams_removed = 0
    horizontal_seams_removed = 0
    
    while np.any(mask):
        energy = compute_energy(img)
        
        # Modify energy to prioritize removing mask pixels
        # Set very low energy for mask pixels so they're preferentially removed
        modified_energy = energy - (mask * 1e6)
        
        # Find the seam that removes the most mask pixels
        (seam, direction), mask_pixels_removed = find_seam_to_remove_mask(modified_energy, mask)
        
        if direction == 'vertical':
            img = remove_vertical_seam(img, seam)
            # Update mask by removing the same seam
            mask = remove_vertical_seam(mask, seam)
            vertical_seams_removed += 1
        else:
            img = remove_horizontal_seam(img, seam)
            # Update mask by removing the same seam
            mask = remove_horizontal_seam(mask, seam)
            horizontal_seams_removed += 1
        
        print(f"vertical: {vertical_seams_removed}; horizontal: {horizontal_seams_removed}")
    
    return img, (vertical_seams_removed, horizontal_seams_removed)

def restore_image_size(image, original_size, seams_removed):
    """
    Restore the image to its original size by inserting seams.
    
    :param image: Image with object removed (numpy array)
    :param original_size: (width, height) of original image
    :param seams_removed: (vertical_seams, horizontal_seams) that were removed
    :return: Image restored to original size
    """
    img = image.copy()
    vertical_seams, horizontal_seams = seams_removed
    
    # Add back vertical seams
    for _ in range(vertical_seams):
        energy = compute_energy(img)
        seam = find_vertical_seam(energy)
        img = add_vertical_seam(img, seam)
    
    # Add back horizontal seams
    for _ in range(horizontal_seams):
        energy = compute_energy(img)
        seam = find_horizontal_seam(energy)
        img = add_horizontal_seam(img, seam)
    
    return img

def object_removal(input_path, output_path, mask_path=None):
    """
    Main function for object removal using seam carving.
    
    :param input_path: Path to input image
    :param output_path: Path to save output image
    :param mask_path: Optional path to mask image
    """
    # Load image
    image = Image.open(input_path)
    img_array = np.array(image)
    original_size = img_array.shape[1], img_array.shape[0]  # (width, height)
    
    # Get or create mask
    mask = create_mask_from_user(input_path, mask_path)
    if mask is None:
        return
    
    # Remove object by removing seams that cover the mask
    img_removed, seams_removed = remove_object_with_mask(img_array, mask)
    Image.fromarray(np.uint8(img_removed)).save("seam_carving\output\couple_middle.png")
    
    # Restore original image size
    img_restored = restore_image_size(img_removed, original_size, seams_removed)
    
    # Save result
    Image.fromarray(np.uint8(img_restored)).save(output_path)
    print(f"Object removed and image saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Object Removal using Seam Carving')
    parser.add_argument('input', type=str, help='Input image file path')
    parser.add_argument('output', type=str, help='Output image file path')
    parser.add_argument('--mask', type=str, help='Optional path to existing mask image',
                       required=False)
    
    args = parser.parse_args()
    
    object_removal(args.input, args.output, args.mask)
    
# python seam_carving\object_removal.py seam_carving\input\couple.png seam_carving\output\couple.png --mask seam_carving\mask\couple_mask.png
