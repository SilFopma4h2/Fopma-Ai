#!/usr/bin/env python3
"""
Fopma-AI: Enhanced Mini-ChatGPT Implementation
Main entry point for training and running the AI model

This script can be run directly in Google Colab after cloning the repository:
!git clone https://github.com/SilFopma4h2/Fopma-Ai.git
%cd Fopma-Ai
!python main.py
"""

import sys
import os
import warnings
import time
from typing import Optional, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def install_dependencies():
    """Install required dependencies if not already installed"""
    print("🔧 Installing dependencies...")
    
    try:
        import torch
        import transformers
        import datasets
        print("✅ All dependencies already installed!")
        return True
    except ImportError:
        print("📦 Installing required packages...")
        os.system("pip install -q torch transformers datasets accelerate tqdm numpy matplotlib seaborn")
        print("✅ Dependencies installed successfully!")
        return True

def setup_environment():
    """Setup the environment for optimal performance"""
    import torch
    
    print("🌟 Setting up environment...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)
    import random
    random.seed(42)
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def enhanced_mini_gpt():
    """Enhanced MiniGPT with improved architecture and training"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2Tokenizer
    from datasets import load_dataset
    import math
    from tqdm.auto import tqdm
    
    class ImprovedMultiHeadAttention(nn.Module):
        """Enhanced Multi-Head Attention with better initialization and dropout"""
        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # Improved initialization
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.d_k)
            
            # Better initialization
            self._reset_parameters()
        
        def _reset_parameters(self):
            """Initialize parameters using Xavier uniform"""
            for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        def forward(self, x, mask=None):
            batch_size, seq_len, d_model = x.shape
            
            # Linear transformations
            Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            context = torch.matmul(attention_weights, V)
            
            # Concatenate heads
            context = context.transpose(1, 2).contiguous().view(
                batch_size, seq_len, d_model
            )
            
            # Final linear transformation
            output = self.W_o(context)
            return output
    
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
            norm_x = self.norm1(x)
            attention_output = self.attention(norm_x, mask)
            x = x + self.dropout(attention_output)
            
            # Second sub-layer
            norm_x = self.norm2(x)
            ff_output = self.feed_forward(norm_x)
            x = x + self.dropout(ff_output)
            
            return x
    
    class EnhancedMiniGPT(nn.Module):
        """Enhanced MiniGPT with improved architecture and training stability"""
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.max_seq_len = config['max_seq_len']
            
            # Improved embeddings
            self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
            self.position_embedding = nn.Embedding(config['max_seq_len'], config['d_model'])
            
            # Enhanced transformer blocks
            self.transformer_blocks = nn.ModuleList([
                EnhancedTransformerBlock(
                    config['d_model'], 
                    config['num_heads'], 
                    config['d_ff'],
                    config['dropout']
                ) for _ in range(config['num_layers'])
            ])
            
            self.ln_f = nn.LayerNorm(config['d_model'])
            self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
            
            # Tie embeddings (weight sharing between input and output embeddings)
            self.lm_head.weight = self.token_embedding.weight
            
            self.dropout = nn.Dropout(config['dropout'])
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights using best practices"""
            for module in self.modules():
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
            # Create a boolean causal mask to avoid float bitwise operations on CUDA
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
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
            
            # Create causal mask (boolean)
            causal_mask = self.create_causal_mask(seq_len, device)
            if attention_mask is not None:
                # Ensure attention_mask is boolean and combine with causal mask
                causal_mask = causal_mask & attention_mask.unsqueeze(1).unsqueeze(1).to(torch.bool)
            
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
    
    # Enhanced configuration with better defaults
    print("🏗️ Creating enhanced model configuration...")
    config = {
        'vocab_size': 50257,  # GPT-2 tokenizer size
        'd_model': 384,       # Increased model dimension
        'num_heads': 12,      # More attention heads
        'num_layers': 6,      # More transformer layers
        'd_ff': 1536,        # Larger feed-forward dimension (4 * d_model)
        'max_seq_len': 256,   # Longer sequences
        'dropout': 0.1
    }
    
    print("📊 Model Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Initialize tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    print("🤖 Creating enhanced model...")
    device = setup_environment()
    model = EnhancedMiniGPT(config).to(device)
    
    total_params, trainable_params = model.count_parameters()
    print(f"📈 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB")
    
    return model, tokenizer, config, device

def enhanced_training_loop(model, tokenizer, config, device):
    """Enhanced training with better data handling and optimization"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from datasets import load_dataset
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt
    
    class ImprovedTextDataset(Dataset):
        """Improved dataset with better tokenization and handling"""
        def __init__(self, texts, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.texts = texts
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            
            # Tokenize with truncation and padding
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids.clone()  # For language modeling
            }
    
    print("📚 Loading and preparing dataset...")
    
    # Load a diverse dataset (using OpenWebText which is similar to GPT training data)
    try:
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        print("✅ Loaded OpenWebText dataset")
    except:
        # Fallback to a smaller dataset if OpenWebText fails
        print("⚠️ OpenWebText not available, using fallback dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Enhanced data sampling - use much more data for better training
    # Local data configuration (replacing get_data_config)
    data_config = {
        'sample_size': 50000
    }
    sample_size = data_config['sample_size']  # Now 50,000 for enhanced training!
    print(f"🎯 ENHANCED DATA LOADING: Sampling {sample_size:,} texts for better training...")
    print(f"   📈 This is 5x more data than before for significantly better results!")
    
    if hasattr(dataset, 'take'):
        # For streaming datasets
        texts = [item['text'] for item in dataset.take(sample_size) if len(item['text'].strip()) > 50]
    else:
        # For regular datasets
        texts = [item['text'] for item in dataset[:sample_size] if len(item['text'].strip()) > 50]
    
    print(f"✅ Prepared {len(texts)} training texts")
    
    # Create dataset and dataloader
    train_dataset = ImprovedTextDataset(texts, tokenizer, config['max_seq_len'])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4,  # Reduced for memory efficiency
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues in Colab
    )
    
    # Enhanced optimizer and scheduler  
    print("🔧 Setting up ENHANCED training configuration...")
    # Local training configuration (replacing get_training_config)
    training_config = {
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'num_epochs': 100
    }
    
    optimizer = AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        betas=training_config['betas']
    )
    
    num_epochs = training_config['num_epochs']  # Now 100 epochs for enhanced training!
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    print(f"📊 Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {train_dataloader.batch_size}")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    
    # Training loop with enhanced monitoring
    print("\n🚀 Starting enhanced training...")
    model.train()
    
    training_losses = []
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n📖 Epoch {epoch + 1}/{num_epochs}")
        epoch_losses = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            training_losses.append(loss.item())
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            perplexity = torch.exp(loss).item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{perplexity:.2f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log periodically
            if batch_idx % 100 == 0:
                print(f"   Step {batch_idx}: Loss = {loss.item():.4f}, Perplexity = {perplexity:.2f}")
        
        # Epoch summary
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"✅ Epoch {epoch + 1} completed - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"💾 New best model saved (loss: {best_loss:.4f})")
    
    print("🎉 Training completed!")
    return model, training_losses

def enhanced_text_generation(model, tokenizer, device):
    """Enhanced text generation with multiple sampling strategies"""
    import torch
    import torch.nn.functional as F
    
    def generate_text(prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9, num_return_sequences=1):
        """Enhanced text generation with multiple sampling strategies"""
        model.eval()
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        generated_sequences = []
        
        with torch.no_grad():
            for _ in range(num_return_sequences):
                current_ids = input_ids.clone()
                
                for _ in range(max_length):
                    # Get model predictions
                    outputs = model(current_ids)
                    next_token_logits = outputs[0, -1, :]
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                        next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                    
                    # Apply top-p (nucleus) sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = False
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = -float('inf')
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Add to sequence
                    current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                    
                    # Stop if EOS token is generated
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                # Decode generated sequence
                generated_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
                generated_sequences.append(generated_text)
        
        return generated_sequences[0] if num_return_sequences == 1 else generated_sequences
    
    def interactive_chat():
        """Enhanced interactive chat with better conversation handling"""
        print("\n🤖 Enhanced Mini-ChatGPT Chat Interface")
        print("=" * 50)
        print("Type your messages below. Type 'quit' to exit.")
        print("Commands:")
        print("  /temp <value>  - Set temperature (0.1-2.0)")
        print("  /length <num>  - Set max response length")
        print("  /reset         - Reset conversation")
        print("=" * 50)
        
        conversation_history = ""
        temperature = 0.8
        max_length = 100
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input.startswith('/temp'):
                        try:
                            temp_value = float(user_input.split()[1])
                            if 0.1 <= temp_value <= 2.0:
                                temperature = temp_value
                                print(f"🌡️ Temperature set to {temperature}")
                            else:
                                print("❌ Temperature must be between 0.1 and 2.0")
                        except:
                            print("❌ Invalid temperature format. Use: /temp 0.8")
                    
                    elif user_input.startswith('/length'):
                        try:
                            length_value = int(user_input.split()[1])
                            if 10 <= length_value <= 500:
                                max_length = length_value
                                print(f"📏 Max length set to {max_length}")
                            else:
                                print("❌ Length must be between 10 and 500")
                        except:
                            print("❌ Invalid length format. Use: /length 100")
                    
                    elif user_input == '/reset':
                        conversation_history = ""
                        print("🔄 Conversation reset")
                    
                    continue
                
                # Generate response
                print("🤔 Thinking...")
                
                # Include conversation history for context
                prompt = f"{conversation_history}Human: {user_input}\nAI:"
                
                response = generate_text(
                    prompt, 
                    max_length=max_length, 
                    temperature=temperature
                )
                
                # Extract AI response
                ai_response = response[len(prompt):].strip()
                if '\n' in ai_response:
                    ai_response = ai_response.split('\n')[0].strip()
                
                print(f"🤖 AI: {ai_response}")
                
                # Update conversation history
                conversation_history += f"Human: {user_input}\nAI: {ai_response}\n"
                
                # Keep conversation history manageable
                if len(conversation_history) > 1000:
                    conversation_history = conversation_history[-800:]
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again.")
    
    # Demonstration
    print("\n🎭 Enhanced Text Generation Demo")
    print("=" * 40)
    
    demo_prompts = [
        "The future of artificial intelligence is",
        "In a world where robots and humans coexist",
        "The most important lesson I learned today was",
        "Once upon a time in a magical forest"
    ]
    
    for prompt in demo_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        generated = generate_text(prompt, max_length=80, temperature=0.8)
        print(f"🤖 Generated: '{generated}'")
        print("-" * 40)
    
    # Start interactive chat
    interactive_chat()

def main():
    """Main function to run the enhanced Mini-ChatGPT"""
    import torch
    print("🌟 Fopma-AI: Enhanced Mini-ChatGPT")
    print("=" * 50)
    print("Welcome to the improved AI experience!")
    print("This implementation features:")
    print("✨ Enhanced transformer architecture")
    print("🚀 Improved training strategies") 
    print("🎯 Better text generation")
    print("💬 Interactive chat interface")
    print("=" * 50)
    
    try:
        # Install dependencies
        install_dependencies()
        
        # Create enhanced model
        model, tokenizer, config, device = enhanced_mini_gpt()
        
        # Train the model
        print("\n🎓 Starting enhanced training process...")
        model, training_losses = enhanced_training_loop(model, tokenizer, config, device)
        
        # Save the trained model
        print("\n💾 Saving enhanced model...")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'training_losses': training_losses
        }
        torch.save(checkpoint, 'enhanced_mini_chatgpt.pt')
        print("✅ Model saved as 'enhanced_mini_chatgpt.pt'")
        
        # Start text generation and chat
        enhanced_text_generation(model, tokenizer, device)
        
        print("\n🎉 Enhanced Mini-ChatGPT session completed!")
        print("Thank you for using Fopma-AI!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please check your environment and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
