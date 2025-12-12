#!/usr/bin/env python3

import torch
from torchvision import transforms
import numpy as np

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_transforms(name='train', image_size=288):
    """
    Get transforms for data augmentation
    
    Args:
        name: 'train' or 'val'
        image_size: size of the input images
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if name == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif name == 'val':
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError('Invalid transform name: {}'.format(name))

