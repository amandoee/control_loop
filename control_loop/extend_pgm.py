#!/usr/bin/env python3
"""
PGM P5 Map Extender
Extends a PGM P5 format image of any size to exactly 1600x1600 pixels.
"""

from PIL import Image
import numpy as np
import argparse

def extend_pgm_to_1600x1600(input_path, output_path, background_value=0):
    """
    Extends a PGM P5 image to 1600x1600 pixels, placing the original image in the center.
    
    Args:
        input_path (str): Path to the input PGM P5 file
        output_path (str): Path to save the output PGM P5 file
        background_value (int): Grayscale value for the background (0-255)
    """
    # Open the PGM image using Pillow
    img = Image.open(input_path)
    
    # Ensure the image is in grayscale mode
    if img.mode != 'L':
        img = img.convert('L')
    
    # Get original dimensions
    width, height = img.size
    
    # Create a new 1600x1600 image with the background color
    new_img = Image.new('L', (1600, 1600), background_value)
    
    # Calculate position to paste the original (centered)
    paste_position = ((1600 - width) // 2, (1600 - height) // 2)
    
    # Paste the original image onto the new canvas
    new_img.paste(img, paste_position)
    
    # Convert to numpy array for custom PGM P5 writing
    new_img_data = np.array(new_img)
    
    # Write as PGM P5 format
    write_pgm_p5(output_path, new_img_data)

def write_pgm_p5(file_path, img_data, max_val=255):
    """
    Writes a numpy array as a PGM P5 (binary) format file.
    
    Args:
        file_path (str): Path to save the PGM P5 file
        img_data (numpy.ndarray): 2D array of grayscale values
        max_val (int): Maximum grayscale value (typically 255)
    """
    with open(file_path, 'wb') as f:
        # Write header
        f.write(b'P5\n')
        f.write(f"{img_data.shape[1]} {img_data.shape[0]}\n".encode())
        f.write(f"{max_val}\n".encode())
        
        # Write image data
        f.write(img_data.tobytes())

def main():
    parser = argparse.ArgumentParser(description='Extend a PGM P5 image to 1600x1600 pixels')
    parser.add_argument('input', help='Input PGM P5 file path')
    parser.add_argument('output', help='Output PGM P5 file path')
    parser.add_argument('--bg', type=int, default=0, 
                        help='Background grayscale value (0-255, default: 0)')
    
    args = parser.parse_args()
    
    # Validate background value
    if args.bg < 0 or args.bg > 255:
        parser.error("Background value must be between 0 and 255")
    
    extend_pgm_to_1600x1600(args.input, args.output, args.bg)
    print(f"Extended image saved to {args.output}")

if __name__ == "__main__":
    main()
