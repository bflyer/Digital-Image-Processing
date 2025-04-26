import numpy as np
from PIL import Image
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

def find_seam_to_remove_mask(energy, mode="vertical"):
    """
    Find the seam that removes the most mask pixels.
    
    :param energy: Energy map (H, W)
    :param mask: Binary mask (H, W) where 1=remove, 0=keep
    :return: (seam, mask_pixels_removed), where seam is vertical or horizontal
    """
    # Find vertical seam that removes most mask pixels
    if mode == "vertical":
        return (find_vertical_seam(energy)[0], 'vertical')
    elif mode == "horizontal":
        return (find_horizontal_seam(energy)[0], 'horizontal')

def remove_object_with_mask(image, mask_preserve, mask_remove, mode='vertical'):
    """
    Remove the masked object by repeatedly removing seams that cover the most mask pixels.
    
    :param image: Input image (numpy array)
    :param mask: Binary mask (H, W) where 1=remove, 0=keep
    :return: Image with object removed, and number of seams removed in each direction
    """
    img = image.copy()
    mask_preserve = mask_preserve.copy()
    mask_remove = mask_remove.copy()
    
    # Count how many seams we remove in each direction
    n_seams_removed = 0
    
    if mode == 'vertical':
        while np.any(mask_remove):
            energy = compute_energy(img)
        
            # Modify energy to prioritize removing mask pixels
            # Set very low energy for mask pixels so they're preferentially removed
            modified_energy = energy - (mask_remove * 1e6) + (mask_preserve * 1e4)
            
            seam = find_vertical_seam(modified_energy)[0]
            img = remove_vertical_seam(img, seam)
            # Update mask by removing the same seam
            mask_preserve = remove_vertical_seam(mask_preserve, seam)
            mask_remove = remove_vertical_seam(mask_remove, seam)
            n_seams_removed += 1
            
            print(f"vertical: {n_seams_removed}")
                
    if mode == 'horizontal':     
        while np.any(mask_remove):
            energy = compute_energy(img)
            
            # Modify energy to prioritize removing mask pixels
            # Set very low energy for mask pixels so they're preferentially removed
            modified_energy = energy - (mask_remove * 1e6) + (mask_preserve * 1e6)

            seam = find_horizontal_seam(modified_energy)[0]
            img = remove_horizontal_seam(img, seam)
            # Update mask by removing the same seam
            mask_preserve = remove_horizontal_seam(mask_preserve, seam)
            mask_remove = remove_horizontal_seam(mask_remove, seam)
            n_seams_removed += 1
            
            print(f"horizontal: {n_seams_removed}")
    
    return img, n_seams_removed

def restore_image_size(image, n_seams_removed, mode='vertical'):
    """
    Restore the image to its original size by inserting seams.
    
    :param image: Image with object removed (numpy array)
    :param n_seams_removed: (vertical_seams, horizontal_seams) that were removed
    :return: Image restored to original size
    """
    img = image.copy()
    
    # Add back vertical seams
    if mode == 'vertical' and n_seams_removed > 0:
        energy = compute_energy(img)
        seams = find_vertical_seam(energy, n_seams_removed)
        img = add_vertical_seam(img, seams)
    
    # Add back horizontal seams
    if mode == 'horizontal' and n_seams_removed > 0:
        energy = compute_energy(img)
        seams = find_horizontal_seam(energy, n_seams_removed)
        img = add_horizontal_seam(img, seams)
    
    return img

def object_removal(input_path, output_path, mask_preserve=None, mask_remove=None, mode='vertical'):
    """
    Main function for object removal using seam carving.
    
    :param input_path: Path to input image
    :param output_path: Path to save output image
    :param mask_path: Optional path to mask image
    """
    # Load image
    image = Image.open(input_path)
    img_array = np.array(image)
    
    # Get or create mask
    mask_preserve = create_mask_from_user(input_path, mask_preserve)
    mask_remove = create_mask_from_user(input_path, mask_remove)
    if mask_preserve is None and mask_remove is None:
        return
    
    # Remove object by removing seams that cover the mask
    img_removed, n_seams_removed = remove_object_with_mask(img_array, mask_preserve, mask_remove, mode)
    Image.fromarray(np.uint8(img_removed)).save("seam_carving\output\couple_middle.png")
    
    # Restore original image size
    img_restored = restore_image_size(img_removed, n_seams_removed, mode)
    
    # Save result
    Image.fromarray(np.uint8(img_restored)).save(output_path)
    print(f"Object removed and image saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Object Removal using Seam Carving')
    parser.add_argument('input', type=str, help='Input image file path')
    parser.add_argument('output', type=str, help='Output image file path')
    parser.add_argument('--mask_preserve', type=str, help='Optional path to existing mask image')
    parser.add_argument('--mask_remove', type=str, help='Optional path to existing mask image')
    parser.add_argument('--mode', type=str, default='vertical', choices=['vertical', 'vertical', 'horizontal'],
                       help='Mode for removing object (default: vertical)')
    
    args = parser.parse_args()
    
    object_removal(args.input, args.output, args.mask_preserve, args.mask_remove, args.mode)
    
# python seam_carving\object_removal.py\
# seam_carving\input\couple.png\
# seam_carving\output\couple.png\
# --mask_preserve seam_carving\mask\mask_preserve.png\
# --mask_remove seam_carving\mask\mask_delete.png\
# --mode vertical 

# python seam_carving\object_removal.py seam_carving\input\couple.png seam_carving\output\couple.png --mask_preserve seam_carving\mask\mask_preserve.png --mask_remove seam_carving\mask\mask_delete.png --mode vertical 
