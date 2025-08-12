import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np


class Augmentation:
    def __init__(self):
        self.train_transform = A.Compose([
            # Apply geometric and elastic deformations
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=25, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),

            # Apply brightness, contrast, and color jitter
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ColorJitter(hue=0.05, saturation=0.1, p=0.3),

            # Add gamma correction for illumination normalization
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),

            # Add occlusion and distortion
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, fill_value=0, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),

            # Add noise and blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),

            # Normalize using ImageNet statistics
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            # Convert to tensor
            ToTensorV2()
        ])

        self.test_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def apply_train(self, img, mask):
        # Apply training-time augmentation
        transformed = self.train_transform(image=img, mask=mask)
        return transformed["image"], transformed["mask"]

    def apply_test(self, img):
        # Apply test-time transformation (normalization only)
        transformed = self.test_transform(image=img)
        return transformed["image"]


if __name__ == "__main__":
    # Paths to your test image and mask
    img_path = "test_data/images/11.png"
    mask_path = "test_data/labels/11.png"

    # Read image and mask
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Initialize augmentation
    aug = Augmentation()

    # Apply train-time augmentation
    aug_img, aug_mask = aug.apply_train(img, mask)
    aug_img = aug_img.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
    aug_mask = aug_mask.cpu().numpy()

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Augmented Image")
    plt.imshow((aug_img * 0.5 + 0.5).clip(0, 1))  # Undo normalization for display
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Augmented Mask")
    plt.imshow(aug_mask.squeeze(), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
