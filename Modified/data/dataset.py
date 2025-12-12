#!/usr/bin/env python3

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    """Custom dataset for loading images with pseudo labels"""
    
    def __init__(self, pseudo_label_dict, data_folder, transform=None):
        """
        Args:
            pseudo_label_dict: Dictionary with image names as keys and pseudo labels as values
            data_folder: Path to folder containing images
            transform: Optional transform to be applied on the images
        """
        self.pseudo_label_dict = pseudo_label_dict
        self.data_folder = data_folder
        self.transform = transform
        
        # Create list of image names and labels
        self.image_names = list(pseudo_label_dict.keys())
        self.labels = [int(pseudo_label_dict[name]) for name in self.image_names]
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.image_names[idx]
        img_path = os.path.join(self.data_folder, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image if there's an error
            image = Image.new('RGB', (288, 288), (0, 0, 0))
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label, img_name

