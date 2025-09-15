"""
Enhanced training implementation with improved optimization strategies
"""

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

try:
    from transformers import AdamW, get_linear_schedule_with_warmup
except ImportError:
    # For newer versions of transformers, AdamW is in torch.optim
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup


class EnhancedTrainer:
    """Enhanced trainer with better optimization and monitoring"""
    
    def __init__(self, model, tokenizer, device, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or self._get_default_config()
        
        # Training state
        self.training_losses = []
        self.best_loss = float('inf')
        
    def _get_default_config(self):
        """Default training configuration"""
        return {
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'betas': (0.9, 0.95),
            'num_epochs': 3,
            'warmup_ratio': 0.1,
            'gradient_clipping': 1.0,
            'save_steps': 500,
            'logging_steps': 10
        }
    
    def setup_optimization(self, dataloader):
        """Setup optimizer and learning rate scheduler"""
        print("🔧 Setting up enhanced optimization...")
        
        # Enhanced optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=self.config['betas']
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(dataloader) * self.config['num_epochs']
        warmup_steps = int(self.config['warmup_ratio'] * total_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"   Learning rate: {self.config['learning_rate']}")
        print(f"   Total steps: {total_steps}")
        print(f"   Warmup steps: {warmup_steps}")
        print(f"   Weight decay: {self.config['weight_decay']}")
    
    def train(self, dataloader):
        """Enhanced training loop with better monitoring for 100 epochs"""
        print(f"🚀 Starting ENHANCED RETRAINING for {self.config['num_epochs']} epochs...")
        print(f"📊 Training Configuration Summary:")
        print(f"   🔄 Epochs: {self.config['num_epochs']} (ENHANCED for better results)")
        print(f"   📦 Batch size: {dataloader.batch_size}")
        print(f"   📚 Total batches per epoch: {len(dataloader)}")
        print(f"   🎯 Total training steps: {len(dataloader) * self.config['num_epochs']:,}")
        print(f"   ⏱️  Estimated training time: ~{(len(dataloader) * self.config['num_epochs'] * 0.5 / 60):.1f} minutes")
        print()
        
        # Setup optimization
        self.setup_optimization(dataloader)
        
        # Training loop
        self.model.train()
        global_step = 0
        
        # Progress tracking for 100 epochs
        epoch_milestones = [10, 25, 50, 75, 90, 100]
        
        for epoch in range(self.config['num_epochs']):
            epoch_losses = []
            
            # Special milestone reporting
            if (epoch + 1) in epoch_milestones:
                print(f"\n🎯 MILESTONE: Epoch {epoch + 1}/{self.config['num_epochs']} - {((epoch + 1)/self.config['num_epochs']*100):.0f}% complete!")
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss (language modeling loss)
                loss = self._calculate_language_modeling_loss(logits, labels, attention_mask)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['gradient_clipping'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clipping']
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Logging
                loss_value = loss.item()
                epoch_losses.append(loss_value)
                self.training_losses.append(loss_value)
                
                # Update progress bar with enhanced info
                current_lr = self.scheduler.get_last_lr()[0]
                avg_loss_recent = sum(self.training_losses[-10:]) / min(10, len(self.training_losses))
                progress_bar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{avg_loss_recent:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': f'{global_step:,}'
                })
                
                # Enhanced periodic logging
                if global_step % self.config['logging_steps'] == 0 and global_step > 0:
                    avg_loss = sum(self.training_losses[-self.config['logging_steps']:]) / self.config['logging_steps']
                    print(f"   📈 Step {global_step:,}: avg_loss={avg_loss:.4f}, lr={current_lr:.2e}")
                
                global_step += 1
            
            # Enhanced epoch summary
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            improvement = ""
            if epoch > 0:
                prev_epoch_start = max(0, len(self.training_losses) - len(epoch_losses) * 2)
                prev_epoch_end = len(self.training_losses) - len(epoch_losses)
                if prev_epoch_end > prev_epoch_start:
                    prev_avg = sum(self.training_losses[prev_epoch_start:prev_epoch_end]) / (prev_epoch_end - prev_epoch_start)
                    change = ((avg_epoch_loss - prev_avg) / prev_avg) * 100
                    if change < 0:
                        improvement = f" (⬇ {abs(change):.1f}% improvement!)"
                    else:
                        improvement = f" (⬆ {change:.1f}% higher)"
            
            print(f"✅ Epoch {epoch + 1}/{self.config['num_epochs']} completed - Average loss: {avg_epoch_loss:.4f}{improvement}")
            
            # Save best model
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self._save_checkpoint(epoch, avg_epoch_loss, 'best_model.pt')
                print(f"   💾 New best model saved! Loss: {avg_epoch_loss:.4f}")
            
            # Save regular checkpoints every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_name = f'checkpoint_epoch_{epoch+1}.pt'
                self._save_checkpoint(epoch, avg_epoch_loss, checkpoint_name)
                print(f"   💾 Checkpoint saved: {checkpoint_name}")
        
        print("\n🎉 ENHANCED RETRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Training Summary:")
        print(f"   🔄 Total epochs completed: {self.config['num_epochs']}")
        print(f"   📈 Total training steps: {global_step:,}")
        print(f"   🎯 Best loss achieved: {self.best_loss:.4f}")
        print(f"   📉 Final loss: {self.training_losses[-1]:.4f}")
        print(f"   💾 Best model saved as: best_model.pt")
        
        return self.training_losses
    
    def _calculate_language_modeling_loss(self, logits, labels, attention_mask):
        """Calculate language modeling loss with proper masking"""
        # Shift logits and labels for language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        
        # Flatten for loss calculation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_attention_mask = shift_attention_mask.view(-1)
        
        # Calculate cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits, shift_labels)
        
        # Apply attention mask (only calculate loss on non-padded tokens)
        loss = loss * shift_attention_mask
        loss = loss.sum() / shift_attention_mask.sum()
        
        return loss
    
    def _save_checkpoint(self, epoch, loss, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
            'training_losses': self.training_losses
        }
        torch.save(checkpoint, filename)
        print(f"💾 Checkpoint saved: {filename}")
    
    def plot_training_progress(self):
        """Plot training loss over time"""
        if not self.training_losses:
            print("No training losses to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, alpha=0.7, label='Training Loss')
        
        # Add smoothed line
        if len(self.training_losses) > 10:
            window_size = max(1, len(self.training_losses) // 20)
            smoothed = []
            for i in range(len(self.training_losses)):
                start = max(0, i - window_size)
                end = min(len(self.training_losses), i + window_size + 1)
                smoothed.append(sum(self.training_losses[start:end]) / (end - start))
            plt.plot(smoothed, color='red', linewidth=2, label='Smoothed Loss')
        
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save_final_model(self, filename='enhanced_mini_chatgpt.pt'):
        """Save the final trained model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config if hasattr(self.model, 'config') else None,
            'training_losses': self.training_losses,
            'tokenizer_info': {
                'vocab_size': len(self.tokenizer),
                'model_name': getattr(self.tokenizer, 'name_or_path', 'gpt2')
            }
        }
        torch.save(checkpoint, filename)
        print(f"✅ Final model saved as '{filename}'")
        return filename