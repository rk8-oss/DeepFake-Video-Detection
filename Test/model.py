import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels=768, features=[64, 128, 256, 512]):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(features[0], features[0], 2, stride=1)
        self.layer2 = self._make_layer(features[0], features[1], 2, stride=2)
        self.layer3 = self._make_layer(features[1], features[2], 2, stride=2)
        self.layer4 = self._make_layer(features[2], features[3], 2, stride=2)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        features = []
        
        x = self.initial(x)
        features.append(x)
        
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        return features

class TokenFusion(nn.Module):
    def __init__(self, dim, num_patches):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        
        # Fix: Change fusion_conv to operate on the correct dimensions after transpose
        self.fusion_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.fusion_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x shape: (batch, num_patches, dim)
        x_t = x.transpose(1, 2)  # (batch, dim, num_patches)
        x_fused = self.fusion_conv(x_t).transpose(1, 2)  # (batch, num_patches, dim)
        x_fused = self.fusion_norm(x_fused)
        return x + x_fused

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)



class M2TR(nn.Module):
    def __init__(
        self,
        *,
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Multi-scale feature extractor
        self.feature_extractor = MultiScaleFeatureExtractor(channels)
        
        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            Transpose(1, 2)  # (B, num_patches, dim)
        )
        
        # Token fusion module
        self.token_fusion = TokenFusion(dim, num_patches)
        
        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # Multi-scale feature extraction
        multi_scale_features = self.feature_extractor(img)
        
        # Standard patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add positional embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        # Apply token fusion
        x_patches = x[:, 1:, :]  # Exclude CLS token
        x_patches = self.token_fusion(x_patches)
        x = torch.cat((x[:, 0:1, :], x_patches), dim=1)  # Reattach CLS token
        
        # Process through transformer
        x = self.transformer(x)
        
        # Pool and classify
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        
        return self.mlp_head(x), x  # Return both classification and feature vector

class M2TRECL(nn.Module):
    """M2TR backbone + projection head for enhanced contrastive learning"""
    def __init__(self, out_dim=128, image_size=224):
        super(M2TRECL, self).__init__()
        
        # Initialize M2TR backbone
        self.backbone = M2TR(
            image_size=image_size,
            patch_size=16,
            num_classes=out_dim,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            channels=3
        )
        
        # Get the embedding dimension from the backbone
        dim_mlp = 768  # This should match the 'dim' parameter above
        
        # Add MLP projection head (similar to the original ECL)
        self.backbone.mlp_head = nn.Sequential(
            nn.LayerNorm(dim_mlp),
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        _, feat = self.backbone(x)  # Get the feature vector before classification
        feat_contrast = F.normalize(self.backbone.mlp_head(feat), dim=1)
        return self.backbone.mlp_head(feat), feat_contrast
    
    def init_weights(self):
        # Initialize weights for the transformer layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Special initialization for positional embeddings
        nn.init.normal_(self.backbone.pos_embedding, std=0.02)
        nn.init.normal_(self.backbone.cls_token, std=0.02)
        
        return True

