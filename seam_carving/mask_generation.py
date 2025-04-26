import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import argparse

class InteractiveMaskGenerator:
    def __init__(self, image_path, mask_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.drawing = False
        self.mode = True  # True for drawing mask, False for erasing
        self.brush_size = 10
        self.mask_path = mask_path
        
    def run(self):
        cv2.namedWindow('Object Removal - Draw Mask')
        cv2.setMouseCallback('Object Removal - Draw Mask', self.mouse_callback)
        
        print("Interactive Mask Generation Instructions:")
        print("- Left click & drag: Draw mask (mark object to remove)")
        print("- Right click & drag: Erase mask")
        print("- '+'/'-': Increase/decrease brush size")
        print("- 's': Save mask and proceed")
        print("- 'q': Quit without saving")
        
        while True:
            # Display image with mask overlay
            display = self.image.copy()
            mask_overlay = self.mask.copy()
            mask_overlay = cv2.cvtColor(mask_overlay, cv2.COLOR_GRAY2BGR)
            mask_overlay[:, :, 1:] = 0  # Make mask appear red
            display = cv2.addWeighted(display, 0.7, mask_overlay, 0.3, 0)
            
            # Show brush size
            cv2.putText(display, f"Brush: {self.brush_size}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Mode: Draw" if self.mode else "Mode: Erase", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Object Removal - Draw Mask', display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                return None  # User quit
            elif key == ord('s'):
                # 保存 mask 为文件
                cv2.imwrite(self.mask_path, self.mask)
                print(f"Mask saved as {self.mask_path}")
                return self.mask  # 同时返回 mask 数组
            elif key == ord('m'):
                self.mode = not self.mode  # Toggle draw/erase mode
            elif key == ord('+'):
                self.brush_size = min(50, self.brush_size + 2)
            elif key == ord('-'):
                self.brush_size = max(1, self.brush_size - 2)
        
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.mode = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.mode = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                if self.mode:
                    # Draw mask (object to remove)
                    cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
                else:
                    # Erase mask
                    cv2.circle(self.mask, (x, y), self.brush_size, 0, -1)
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.drawing = False
            
def main():
    parser = argparse.ArgumentParser(description="Interactive Mask Drawing Tool")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", required=False, help="Path to save output mask (PNG)")

    args = parser.parse_args()

    generator = InteractiveMaskGenerator(args.input, args.output)
    generator.run()

if __name__ == "__main__":
    main()
    
# python mask_generation.py --input seam_carving\input\couple.png --output seam_carving\input\mask.png