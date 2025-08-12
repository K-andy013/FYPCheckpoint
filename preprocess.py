import os
import cv2
import numpy as np
from tqdm import tqdm
from woundProcessor import WoundPreprocessor
from augmentation import Augmentation

# Set your input and output directories
input_img_dir = "test_data\images"
input_mask_dir = "test_data\labels"
output_img_dir = "test_data\dir\processed\images"
output_mask_dir = "test_data\dir\processed\labels"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Initialize preprocessor and augmentation
preprocessor = WoundPreprocessor(target_size=(256, 256))
augmentor = Augmentation()

# List all image files
img_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in tqdm(img_files, desc="Preprocessing"):
    img_path = os.path.join(input_img_dir, img_name)
    mask_path = os.path.join(input_mask_dir, img_name)

    # Preprocess (returns preprocessed numpy arrays)
    img, mask, _ = preprocessor(img_path, mask_path)

    # Check if mask is None before using it
    if mask is None:
        print(f"Warning: Mask not found for {img_name}, skipping.")
        continue

    # Ensure uint8 format for Albumentations compatibility later
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    # Save preprocessed images/masks
    cv2.imwrite(os.path.join(output_img_dir, img_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_mask_dir, img_name), mask)

print("Preprocessing complete. All images saved in uint8 format.")
