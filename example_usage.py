#!/usr/bin/env python3
"""
Example usage script for Mini-ChatGPT components
This script demonstrates how to use the model components outside of Colab
"""

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

def create_mini_gpt_example():
    """Create a minimal example of the MiniGPT model"""
    
    print("🤖 Mini-ChatGPT Example Usage")
    print("=" * 40)
    
    # Initialize tokenizer
    print("1. Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   Vocabulary size: {len(tokenizer)}")
    
    # Model configuration (smaller for example)
    config = {
        'vocab_size': len(tokenizer),
        'd_model': 128,      # Smaller for quick example
        'num_heads': 4,
        'num_layers': 2,     # Fewer layers for quick example
        'd_ff': 512,
        'max_seq_len': 64,   # Shorter sequences
        'dropout': 0.1
    }
    
    print(f"\n2. Model configuration:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    # Create a simple version of the attention mechanism
    class SimpleAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            
            self.qkv = nn.Linear(d_model, d_model * 3)
            self.out = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            
            # Simplified attention (not causal for this example)
            att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
            att = torch.softmax(att, dim=-1)
            y = att @ v
            y = y.transpose(1, 2).reshape(B, T, C)
            return self.out(y)
    
    # Simple transformer block
    class SimpleTransformerBlock(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.attention = SimpleAttention(d_model, num_heads)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Linear(4 * d_model, d_model),
            )
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            
        def forward(self, x):
            x = x + self.attention(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x
    
    # Minimal GPT model
    class MiniGPTExample(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            
            self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
            self.position_embedding = nn.Embedding(config['max_seq_len'], config['d_model'])
            
            self.blocks = nn.ModuleList([
                SimpleTransformerBlock(config['d_model'], config['num_heads'])
                for _ in range(config['num_layers'])
            ])
            
            self.ln_f = nn.LayerNorm(config['d_model'])
            self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
            
        def forward(self, x):
            B, T = x.shape
            
            # Token and position embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
            tok_emb = self.token_embedding(x)
            pos_emb = self.position_embedding(pos)
            x = tok_emb + pos_emb
            
            # Transformer blocks
            for block in self.blocks:
                x = block(x)
            
            # Final layer norm and projection
            x = self.ln_f(x)
            logits = self.lm_head(x)
            return logits
    
    print("\n3. Creating model...")
    model = MiniGPTExample(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    print("\n4. Testing model with sample input...")
    
    # Test with sample text
    sample_text = "Hello, this is a test of the mini GPT model."
    input_ids = tokenizer.encode(sample_text, return_tensors='pt')
    print(f"   Input text: '{sample_text}'")
    print(f"   Input shape: {input_ids.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        print(f"   Output shape: {logits.shape}")
        print(f"   Output logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    # Simple generation example
    print("\n5. Simple text generation example...")
    
    def simple_generate(model, tokenizer, prompt, max_new_tokens=10):
        """Simple greedy generation"""
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(input_ids)
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    prompt = "The future of AI"
    generated = simple_generate(model, tokenizer, prompt, max_new_tokens=5)
    print(f"   Prompt: '{prompt}'")
    print(f"   Generated: '{generated}'")
    print("   (Note: Untrained model will produce random/nonsensical output)")
    
    print("\n6. Training example setup...")
    
    # Example training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Dummy training step
    model.train()
    optimizer.zero_grad()
    
    # Create dummy batch
    dummy_input = torch.randint(0, config['vocab_size'], (2, 32))  # batch_size=2, seq_len=32
    dummy_target = torch.randint(0, config['vocab_size'], (2, 32))
    
    logits = model(dummy_input)
    loss = criterion(logits.view(-1, logits.size(-1)), dummy_target.view(-1))
    loss.backward()
    optimizer.step()
    
    print(f"   Training step completed. Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 40)
    print("✅ Mini-ChatGPT example completed!")
    print("\nThis demonstrates the core components used in the Colab notebook:")
    print("- Tokenization with GPT-2 tokenizer")
    print("- Transformer architecture (attention + MLP)")
    print("- Forward pass and text generation")
    print("- Training setup with optimizer and loss")
    print("\nFor the full implementation with training on The Pile dataset,")
    print("use the mini_chatgpt_colab.ipynb notebook in Google Colab!")

if __name__ == "__main__":
    try:
        create_mini_gpt_example()
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install torch transformers")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have PyTorch and transformers installed.")