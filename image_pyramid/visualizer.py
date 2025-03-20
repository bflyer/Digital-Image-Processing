import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import argparse

def show_images(images_path):
    images = []

    # Load images from the given directory
    for filename in os.listdir(images_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(images_path, filename)
            img = Image.open(img_path)
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)

    # Display each image
    for i, img_array in enumerate(images):
        plt.figure(figsize=(10, 10))
        plt.imshow(img_array, cmap='gray')
        plt.title(f"Gaussian Pyramid Level {i+1}")
        plt.axis('off')
        plt.show()

def main(images_path):
    show_images(images_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Images in a Directory')
    parser.add_argument('-p', '--images_path', required=True, help='Path to the directory containing images')
    
    args = parser.parse_args()
    main(args.images_path)