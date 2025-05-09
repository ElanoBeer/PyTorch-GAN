import glob
import random
import os
import numpy as np
import json

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=512, mask_size=128, mode="train", annotation_file=None):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size  # Standard mask size used throughout dataset
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.png" % root))

        # Adjust split ratio for train/validation
        split_ratio = 0.9  # 90% train, 10% validation
        split_idx = int(len(self.files) * split_ratio)
        self.files = self.files[:split_idx] if mode == "train" else self.files[split_idx:]

        # Load bounding box annotations from JSON file
        self.annotations = {}
        if annotation_file:
            try:
                with open(annotation_file, 'r') as f:
                    self.annotations = json.load(f)
                print(f"Successfully loaded annotations for {len(self.annotations)} images")
            except Exception as e:
                print(f"Error loading annotation file: {e}")
                print("Falling back to random masking")

    def apply_bbox_mask(self, img, img_name):
        """Apply mask based on a randomly chosen object bounding box"""
        base_name = os.path.basename(img_name)

        if base_name in self.annotations and "bboxes" in self.annotations[base_name]:
            bboxes = self.annotations[base_name]["bboxes"]

            # If multiple bboxes, randomly select one
            if isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], (list, tuple)):
                bbox = random.choice(bboxes)
            else:
                bbox = bboxes  # single bbox

            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            # # Ensure coordinates are valid
            # x1 = max(0, min(x1, self.img_size - 1))
            # y1 = max(0, min(y1, self.img_size - 1))
            # x2 = max(x1 + 1, min(x2, self.img_size))
            # y2 = max(y1 + 1, min(y2, self.img_size))

            # Center the mask on the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Use standard mask size
            half_size = self.mask_size // 2
            new_x1 = max(0, center_x - half_size)
            new_y1 = max(0, center_y - half_size)

            # Adjust if mask would go beyond image boundaries
            if new_x1 + self.mask_size > self.img_size:
                new_x1 = self.img_size - self.mask_size
            if new_y1 + self.mask_size > self.img_size:
                new_y1 = self.img_size - self.mask_size

            new_x2 = new_x1 + self.mask_size
            new_y2 = new_y1 + self.mask_size

            # Extract the masked part and create the masked image
            masked_part = img[:, new_y1:new_y2, new_x1:new_x2].clone()
            masked_img = img.clone()
            masked_img[:, new_y1:new_y2, new_x1:new_x2] = 1

            return masked_img, masked_part, (new_y1, new_x1)
        else:
            return self.apply_random_mask(img)

    def apply_random_mask(self, img):
        """Randomly masks image"""
        # Ensure mask fits within image bounds
        y1 = random.randint(0, self.img_size - self.mask_size)
        x1 = random.randint(0, self.img_size - self.mask_size)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size

        # Extract the masked part before applying the mask
        masked_part = img[:, y1:y2, x1:x2].clone()
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part, (y1, x1)

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Calculate center coordinates
        i = (self.img_size - self.mask_size) // 2
        j = i  # Same for square images

        # Extract the masked part and create the masked image
        masked_part = img[:, i:i + self.mask_size, j:j + self.mask_size].clone()
        masked_img = img.clone()
        masked_img[:, i:i + self.mask_size, j:j + self.mask_size] = 1

        return masked_img, masked_part, (i, j)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = Image.open(img_path)

        # Convert grayscale images to RGB to ensure 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = self.transform(img)

        if self.annotations:
            # Use bounding box masks when annotations are available
            masked_img, masked_part, (y1, x1) = self.apply_bbox_mask(img, img_path)
        elif self.mode == "train":
            # For training data perform random mask when no annotations are available
            masked_img, masked_part, (y1, x1) = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image when no annotations are available
            masked_img, masked_part, (y1, x1) = self.apply_center_mask(img)

        return img, masked_img, masked_part, (y1, x1)

    def __len__(self):
        return len(self.files)