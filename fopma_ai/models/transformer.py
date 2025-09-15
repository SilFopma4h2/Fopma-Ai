"""
Enhanced Transformer components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import ImprovedMultiHeadAttention


class EnhancedFeedForward(nn.Module):
    """Enhanced Feed-Forward Network with GELU activation and better dropout"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class EnhancedTransformerBlock(nn.Module):
    """Enhanced Transformer Block with pre-normalization and improved residual connections"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = ImprovedMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = EnhancedFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-normalization
        normed_x = self.norm1(x)
        attn_output = self.attention(normed_x, mask)
        x = x + self.dropout(attn_output)
        
        # Pre-normalization for feed-forward
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)
        
        return x