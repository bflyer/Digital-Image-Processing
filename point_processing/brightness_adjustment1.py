from PIL import Image
import sys

def increase_brightness(increment: int, image_path: str, output_path: str = None):
    """
    Increase the brightness of an image.
    
    Parameters:
        increment (int): The amount to increase the brightness by.
        image_path (str): The path to the input image file.
        output_path (str): The path to save the output image file. Defaults to input filename with '_out' appended.
    """
    if not output_path:
        output_path = f"{image_path.rsplit('.', 1)[0]}_out1_increase.{image_path.rsplit('.', 1)[1]}"
    
    with Image.open(image_path) as img:
        enhancer = img.point(lambda p: min(255, p + increment))
        enhancer.save(output_path)

def decrease_brightness(decrement: int, image_path: str, output_path: str = None):
    """
    Decrease the brightness of an image.
    
    Parameters:
        decrement (int): The amount to decrease the brightness by.
        image_path (str): The path to the input image file.
        output_path (str): The path to save the output image file. Defaults to input filename with '_out' appended.
    """
    if not output_path:
        output_path = f"{image_path.rsplit('.', 1)[0]}_out1_decrease.{image_path.rsplit('.', 1)[1]}"
    
    with Image.open(image_path) as img:
        enhancer = img.point(lambda p: max(0, p - decrement))
        enhancer.save(output_path)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <operation> <increment/decrement> <input_file> [output_file]")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    value = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    try:
        if operation == "increase":
            increase_brightness(value, input_file, output_file)
        elif operation == "decrease":
            decrease_brightness(value, input_file, output_file)
        else:
            print("Invalid operation. Please use 'increase' or 'decrease'.")
            sys.exit(1)
        
        print(f"Processed image saved as {output_file}" if output_file else f"{input_file.rsplit('.', 1)[0]}_out1_{operation}.{input_file.rsplit('.', 1)[1]}")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

# python brightness_adjustment1.py increase 30 input.jpg