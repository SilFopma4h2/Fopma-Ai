"""
Enhanced Mini-GPT model implementation
"""

import torch
import torch.nn as nn
from .transformer import EnhancedTransformerBlock


class EnhancedMiniGPT(nn.Module):
    """Enhanced MiniGPT with improved architecture and training stability"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Enhanced embeddings
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embedding = nn.Embedding(config['max_seq_len'], config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])
        
        # Enhanced transformer blocks
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(
                config['d_model'], 
                config['num_heads'], 
                config['d_ff'], 
                config['dropout']
            )
            for _ in range(config['num_layers'])
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # Tie embeddings (common practice in language models)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights with improved strategy"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len, device):
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + position_embeds)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        if attention_mask is not None:
            # Combine with padding mask
            causal_mask = causal_mask & attention_mask.unsqueeze(1).unsqueeze(1)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, causal_mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params