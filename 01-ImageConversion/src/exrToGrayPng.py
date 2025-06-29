import cv2
import numpy as np
import os
import argparse


def compute_global_range(input_dir, clamp_min=0.0, clamp_max=0.85):
    """
    Scans all EXR images in input_dir to find the global min and max
    (after optionally clamping with clamp_min and clamp_max).
    Returns (global_min, global_max).
    """
    global_min = float('inf')
    global_max = float('-inf')

    for filename in os.listdir(input_dir):
        if filename.endswith(".exr"):
            exr_path = os.path.join(input_dir, filename)
            # Read EXR as float32
            img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue  # Skip invalid reads

            # Optional clamp
            img = np.clip(img, clamp_min, clamp_max)

            # Update global min/max
            current_min = img.min()
            current_max = img.max()

            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max

    # In case all images are empty or something goes wrong:
    if global_min == float('inf'):
        global_min = 0.0
    if global_max == float('-inf'):
        global_max = 1.0

    return global_min, global_max


def convert_exr_to_png(exr_path, output_path, global_min, global_max):
    """
    Reads the EXR file as float32, clamps to [global_min, global_max],
    rescales to [0..255], then converts to grayscale and writes as PNG.
    """
    img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Could not read '{exr_path}'. Skipping.")
        return

    # Clamp using the global min/max
    img = np.clip(img, global_min, global_max)

    # Scale to [0..1]
    scaled = (img - global_min) / (global_max - global_min + 1e-12)
    scaled = np.clip(scaled, 0.0, 1.0)

    # Scale to [0..255], then uint8
    scaled_8bit = (scaled * 255).astype(np.uint8)

    # Convert to grayscale
    gray_img = cv2.cvtColor(scaled_8bit, cv2.COLOR_RGB2GRAY)


    cv2.imwrite(output_path, gray_img)
    print(f"Saved {output_path}")



def convert_directory(input_dir, output_dir):
    """
    1) Compute global min/max over all EXRs in 'input_dir'.
    2) Convert each image using that global range, saving to 'output_dir'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Computing global min/max across all EXR files...")
    global_min, global_max = compute_global_range(input_dir, clamp_min=0.0, clamp_max=0.85)
    print(f"  Global min: {global_min}\n  Global max: {global_max}\n")

    print("Converting EXR files using the computed global range...")
    for filename in os.listdir(input_dir):
        if filename.endswith(".exr"):
            exr_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_filename)
            convert_exr_to_png(exr_path, output_path, global_min, global_max)


def main():
    parser = argparse.ArgumentParser(
        description="Convert EXR images to grayscale PNG with global brightness consistency.")
    parser.add_argument('--file', type=str, required=True,
                        help="Path to the folder containing on_centre or off_centre data (expects 'im/rgb' inside).")

    args = parser.parse_args()
    # Input/Output folders
    input_folder = os.path.join(args.file, r"im", "rgb")
    output_folder = os.path.join(args.file, r"im", "gray")

    convert_directory(input_folder, output_folder)


if __name__ == "__main__":
    main()
