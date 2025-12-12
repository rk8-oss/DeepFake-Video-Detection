# !/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class ECLoss(nn.Module):
    """Enhanced Contrastive Loss function for the ECL model"""
    def __init__(self, temperature=0.07):
        super(ECLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        """
        Args:
            features: feature vectors from the contrastive network [batch_size, n_views, dim]
            labels: ground truth of shape [batch_size]
        Returns:
            A loss scalar.
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Gather all features
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # Mask for positive pair similarity
        mask = torch.zeros((batch_size * 2, batch_size * 2), dtype=torch.float32, device=device)
        
        # Set positive pair masks based on labels
        for i in range(batch_size):
            mask[i, batch_size + i] = 1.0
            mask[batch_size + i, i] = 1.0
            
            # Find samples with the same label
            same_label_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
            for j in same_label_indices:
                if i != j:
                    mask[i, j] = 0.5  # Partial positive weight for same class
                    mask[i, batch_size + j] = 0.5
                    mask[batch_size + i, j] = 0.5
                    mask[batch_size + i, batch_size + j] = 0.5
        
        # Compute logits
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature
        
        # Remove self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size * 2).view(-1, 1).to(device), 
            0
        )
        
        # Compute log_prob
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss
       
       