import os
import cv2
import numpy as np
from augmentation import Augmentation  # Make sure augmentation.py is in the same directory or in PYTHONPATH
import shutil

def augment_folder(img_dir, mask_dir, out_img_dir, out_mask_dir, n_aug=3):
    """
    Apply augmentations to all images and masks in a folder and save the results.
    Also copies the original image and mask to the output folders.
    n_aug: number of augmented versions per image
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    aug = Augmentation()

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"Skipping {img_file}: image or mask not found.")
            continue

        # Save original image and mask
        orig_img_out = os.path.join(out_img_dir, f"{os.path.splitext(img_file)[0]}_orig.png")
        orig_mask_out = os.path.join(out_mask_dir, f"{os.path.splitext(img_file)[0]}_orig.png")
        cv2.imwrite(orig_img_out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(orig_mask_out, mask)

        for i in range(n_aug):
            aug_img, aug_mask = aug.apply_train(img, mask)
            # Convert tensors to numpy arrays
            aug_img_np = aug_img.permute(1, 2, 0).cpu().numpy()
            aug_mask_np = aug_mask.cpu().numpy()
            # Undo normalization for saving
            aug_img_np = (aug_img_np * 0.229 + 0.485) * 255  # Approximate, adjust if needed
            aug_img_np = np.clip(aug_img_np, 0, 255).astype(np.uint8)
            aug_mask_np = np.clip(aug_mask_np, 0, 255).astype(np.uint8)
            # Save
            out_img_path = os.path.join(out_img_dir, f"{os.path.splitext(img_file)[0]}_aug{i}.png")
            out_mask_path = os.path.join(out_mask_dir, f"{os.path.splitext(img_file)[0]}_aug{i}.png")
            cv2.imwrite(out_img_path, cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_mask_path, aug_mask_np)
        print(f"Augmented {img_file}")

# Example usage:
if __name__ == "__main__":
    img_dir = "test_data/images"
    mask_dir = "test_data/labels"
    out_img_dir = "test_data/processed/images"
    out_mask_dir = "test_data/processed/labels"
    augment_folder(img_dir, mask_dir, out_img_dir, out_mask_dir, n_aug=3)    
    from augmentation import Augmentation  # Make sure augmentation.py is in the same directory or in PYTHONPATH
    
    def augment_folder(img_dir, mask_dir, out_img_dir, out_mask_dir, n_aug=3):
        """
        Apply augmentations to all images and masks in a folder and save the results.
        n_aug: number of augmented versions per image
        """
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)
        aug = Augmentation()
    
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print(f"Skipping {img_file}: image or mask not found.")
                continue
            for i in range(n_aug):
                aug_img, aug_mask = aug.apply_train(img, mask)
                # Convert tensors to numpy arrays
                aug_img_np = aug_img.permute(1, 2, 0).cpu().numpy()
                aug_mask_np = aug_mask.cpu().numpy()
                # Undo normalization for saving
                aug_img_np = (aug_img_np * 0.229 + 0.485) * 255  # Approximate, adjust if needed
                aug_img_np = np.clip(aug_img_np, 0, 255).astype(np.uint8)
                aug_mask_np = np.clip(aug_mask_np, 0, 255).astype(np.uint8)
                # Save
                out_img_path = os.path.join(out_img_dir, f"{os.path.splitext(img_file)[0]}_aug{i}.png")
                out_mask_path = os.path.join(out_mask_dir, f"{os.path.splitext(img_file)[0]}_aug{i}.png")
                cv2.imwrite(out_img_path, cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR))
                cv2.imwrite(out_mask_path, aug_mask_np)
            print(f"Augmented {img_file}")
    
    # Example usage:
    if __name__ == "__main__":
        img_dir = "test_data/dir/processed/images"
        mask_dir = "test_data/dir/processed/labels"
        out_img_dir = "test_data/augmented/images"
        out_mask_dir = "test_data/augmented/labels"
        augment_folder(img_dir, mask_dir, out_img_dir, out_mask_dir, n_aug=3)