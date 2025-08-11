# -*- coding: utf-8 -*-
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import exposure
import matplotlib.pyplot as plt
import torch

class WoundPreprocessor:
    def __init__(self, target_size=(256, 256), gamma_range=(0.8, 1.5), 
                 use_color_norm=True):
        self.target_size = target_size
        self.gamma_range = gamma_range
        self.use_color_norm = use_color_norm
        
    

    def __call__(self, img_path, mask_path=None):
        """Main preprocessing pipeline"""
        # 1. Load and validate inputs
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
        
        if mask is not None:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 2. Crop black borders
        img, mask = self._crop_black_borders(img, mask)

        # 3. Resize with aspect ratio preservation
        img, mask = self._resize_pad(img, mask)

        # 4. Color normalization (optional)
        if self.use_color_norm:
            img = self._color_staining(img)

        # 5. Adaptive gamma correction
        img = self._adaptive_gamma(img)
        
        # 6. Multi-scale processing
        pyramid = self._build_pyramid(img)
        
        # 7. Intensity normalization
        img = self._percentile_normalization(img, mask)
        return img, mask, pyramid  # <-- Add this line
      

    def _resize_pad(self, img, mask=None):
        """Resize preserving aspect ratio with zero-padding"""
        h, w = img.shape[:2]
        ratio = min(self.target_size[0]/h, self.target_size[1]/w)
        new_size = (int(w*ratio), int(h*ratio))
        
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
        
        # Pad to target size
        delta_w = self.target_size[1] - new_size[0]
        delta_h = self.target_size[0] - new_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=0)
        if mask is not None:
            mask = cv2.copyMakeBorder(mask, top, bottom, left, right, 
                                    cv2.BORDER_CONSTANT, value=0)
        
        return img, mask

    def _color_staining(self, img):
        """Simplified stain normalization"""
        od = -np.log((img.astype(np.float32) + 1) / 255)
        mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)[1]
        background_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        # If too much background, skip normalization
        if background_pixels > 0.8 * total_pixels:
            print("Too much background detected, skipping color normalization.")
            return img
        if np.sum(mask == 0) > 0:
            mean = np.mean(od[mask == 0], axis=0)
            std = np.std(od[mask == 0], axis=0)
            od = (od - mean) / (std + 1e-6)
        
        # Convert back to RGB
        normalized = np.exp(-od)
        alpha = 0.4  # keep more of the normalized, less of the original
        img = cv2.addWeighted(img, 1 - alpha, normalized.astype(np.uint8), alpha, 0)
        return img

    def _adaptive_gamma(self, img):
        """Auto-gamma correction based on image brightness"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mid = 0.5
        mean = np.mean(gray)/255
        gamma = np.log(mid)/np.log(mean) if mean > 0 else 1.0
        gamma = np.clip(gamma, self.gamma_range[0], self.gamma_range[1])
        return exposure.adjust_gamma(img, gamma=gamma)

    def _build_pyramid(self, img, scales=[0.5, 1.0, 1.5]):
        """Create multi-scale pyramid"""
        pyramid = []
        for scale in scales:
            if scale == 1.0:
                pyramid.append(img)
            else:
                h, w = img.shape[:2]
                new_size = (int(w*scale), int(h*scale))
                resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
                # Pad back to original size if needed
                if scale < 1.0:
                    resized = cv2.copyMakeBorder(resized, 
                                              (h-resized.shape[0])//2, 
                                              (h-resized.shape[0]+1)//2,
                                              (w-resized.shape[1])//2,
                                              (w-resized.shape[1]+1)//2,
                                              cv2.BORDER_CONSTANT, value=0)
                pyramid.append(resized)
        return pyramid

    def _percentile_normalization(self, img, mask=None, pmin=1, pmax=99):
        """Robust intensity normalization"""
        if mask is not None and np.sum(mask) > 0:
            pixels = img[mask > 0]
        else:
            pixels = img.reshape(-1, 3)
        
        normalized = np.zeros_like(img, dtype=np.float32)
        for i in range(3):
            channel = pixels[:, i] if len(pixels.shape) > 1 else pixels
            low, high = np.percentile(channel, [pmin, pmax])
            if high <= low:
                # Fallback: use min/max of the image channel
                low, high = np.min(img[..., i]), np.max(img[..., i])
            if high <= low:
                normalized[..., i] = img[..., i]
            else:
                normalized[..., i] = np.clip((img[..., i].astype(np.float32) - low) / (high - low), 0, 1)
        # Debug: print min/max
        print("After normalization, min:", normalized.min(), "max:", normalized.max())
        print("Normalized min:", normalized.min(), "max:", normalized.max())
        return (normalized * 255).astype(np.uint8)
    
    def _crop_black_borders(self, img, mask=None):
        """Crop out black (zero) borders from the image (and mask if provided)."""
        # Create a mask of non-black pixels
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        coords = cv2.findNonZero((gray > 0).astype(np.uint8))
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img_cropped = img[y:y+h, x:x+w]
            if mask is not None:
                mask_cropped = mask[y:y+h, x:x+w]
                return img_cropped, mask_cropped
            return img_cropped, None
        # If all black, return original
        return img, mask

# Example usage:
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = WoundPreprocessor(target_size=(256, 256))
    
    # Process sample image
    img_path = "test_data/image.png"
    mask_path = "test_data/mask.png"

    img, mask,pyramid= preprocessor(img_path, mask_path)
    #add pyramid in above line
    # Visualization
plt.figure(figsize=(12, 6))

# 1. Original image
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
plt.axis('off')

# 2. Processed image
plt.subplot(1, 3, 2)
plt.title("Processed")
if isinstance(img, torch.Tensor):
    img_for_display = img.clone().detach().cpu().permute(1, 2, 0).numpy()
    if img_for_display.min() < 0:
        img_for_display = img_for_display * 0.5 + 0.5  # undo [-1, 1] norm
elif isinstance(img, np.ndarray):
    img_for_display = img.astype(np.float32)
    if img_for_display.max() > 1.0:
        img_for_display /= 255.0
img_for_display = np.clip(img_for_display, 0, 1)
plt.imshow(img_for_display)
plt.axis('off')

# 3. Mask
plt.subplot(1, 3, 3)
plt.title("Mask")
if mask is not None:
    if isinstance(mask, torch.Tensor):
        mask_display = mask.cpu().numpy()
    else:
        mask_display = mask
    plt.imshow(mask_display.squeeze(), cmap='gray')
else:
    plt.text(0.5, 0.5, "No Mask", ha='center', va='center')
plt.axis('off')

plt.tight_layout()
plt.show()
