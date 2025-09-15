#!/usr/bin/env python3
"""
Fopma-AI: Enhanced Mini-ChatGPT Implementation
Main entry point for training and running the AI model

This script can be run directly in Google Colab after cloning the repository:
!git clone https://github.com/SilFopma4h2/Fopma-Ai.git
%cd Fopma-Ai
!python main.py

NOTE: This is now the modular version using improved file structure and training data.
The old monolithic code has been refactored into organized modules for better maintainability.
"""

import sys
import os
import warnings
from typing import Optional, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the current directory to Python path for imports  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def install_dependencies():
    """Install required dependencies if not already installed"""
    from fopma_ai.utils import install_dependencies
    return install_dependencies()

def setup_environment():
    """Setup the environment for optimal performance"""
    from fopma_ai.utils import setup_environment
    return setup_environment()

def enhanced_mini_gpt():
    """Enhanced MiniGPT with improved architecture and training"""
    from fopma_ai.models import EnhancedMiniGPT
    from fopma_ai.utils.config import get_default_config
    from transformers import GPT2Tokenizer
    
    print("🏗️ Creating enhanced model configuration...")
    config = get_default_config()
    
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
    from fopma_ai.data import DataManager
    from fopma_ai.training import EnhancedTrainer
    from fopma_ai.utils.config import get_training_config
    
    # Setup data manager with enhanced datasets
    print("📚 Setting up enhanced data manager...")
    data_manager = DataManager(
        tokenizer=tokenizer,
        max_length=config['max_seq_len'],
        batch_size=4
    )
    
    # Load high-quality training data
    print("🎯 Loading enhanced training datasets...")
    from fopma_ai.utils.config import get_data_config
    data_config = get_data_config()
    texts = data_manager.get_high_quality_datasets(sample_size=data_config['sample_size'])
    
    if not texts:
        print("❌ Failed to load training data")
        return model, []
    
    # Create dataloader
    train_dataloader = data_manager.create_dataloader(texts, shuffle=True)
    print(f"✅ Created training dataloader with {len(train_dataloader)} batches")
    
    # Setup trainer
    print("🎓 Setting up enhanced trainer...")
    training_config = get_training_config()
    trainer = EnhancedTrainer(model, tokenizer, device, training_config)
    
    # Train the model
    print("🚀 Starting enhanced training process...")
    training_losses = trainer.train(train_dataloader)
    
    return model, training_losses

def enhanced_text_generation(model, tokenizer, device):
    """Enhanced text generation and chat interface"""
    from fopma_ai.generation import TextGenerator
    
    # Setup text generator
    print("🎯 Setting up enhanced text generator...")
    generator = TextGenerator(model, tokenizer, device)
    
    # Generate example outputs
    print("\n📝 Generating example outputs...")
    generator.generate_examples([
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The most important lesson I learned today is",
        "When we think about innovation, we should consider"
    ])
    
    # Start interactive chat
    print("\n💬 Starting enhanced chat interface...")
    print("(You can type 'quit' to exit, 'settings' to adjust parameters)")
    generator.chat_interface()

def main():
    """Main function to run the enhanced system"""
    print("🚀" + "="*48 + "🚀")
    print("🤖 FOPMA-AI: Enhanced Mini-ChatGPT")
    print("🚀" + "="*48 + "🚀")
    print()
    print("🌟 Enhanced Features:")
    print("✨ Modular architecture for better maintainability")
    print("🚀 Multiple high-quality training datasets")
    print("🎯 Advanced text generation strategies")
    print("💬 Interactive chat interface")
    print("🔧 Improved optimization and training")
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
        import torch
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