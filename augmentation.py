import albumentations as A
from albumentations.pytorch import ToTensorV2

class Augmentation:
    def __init__(self):
        self.train_transform = A.Compose([
            # Apply geometric and elastic deformations
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=25, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.3),

            # Apply brightness, contrast, and color jitter
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ColorJitter(hue=0.05, saturation=0.1, p=0.3),
            
            # Add gamma correction for illumination normalization
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),  # <--- Added here

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
