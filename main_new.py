#!/usr/bin/env python3
"""
Fopma-AI: Enhanced Mini-ChatGPT Implementation - New Modular Version
Main entry point for training and running the AI model

This script can be run directly in Google Colab after cloning the repository:
!git clone https://github.com/SilFopma4h2/Fopma-Ai.git
%cd Fopma-Ai  
!python main_new.py
"""

import sys
import os
import warnings
from typing import Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function to run the enhanced Fopma-AI system"""
    print("🚀" + "="*48 + "🚀")
    print("🤖 FOPMA-AI: Enhanced Mini-ChatGPT v2.0")
    print("🚀" + "="*48 + "🚀")
    print()
    print("🌟 New Modular Architecture Features:")
    print("✨ Improved file structure and organization") 
    print("🚀 Enhanced training with multiple datasets")
    print("🎯 Better text generation strategies")
    print("💬 Advanced chat interface")
    print("🔧 Automatic system optimization")
    print("=" * 50)
    
    try:
        # Import modules (will install dependencies if needed)
        from fopma_ai.utils import install_dependencies, setup_environment
        from fopma_ai.models import EnhancedMiniGPT
        from fopma_ai.data import DataManager
        from fopma_ai.training import EnhancedTrainer
        from fopma_ai.generation import TextGenerator
        from fopma_ai.utils.config import get_default_config, get_training_config
        from transformers import GPT2Tokenizer
        import torch
        
        # Install dependencies
        install_dependencies()
        
        # Setup environment
        device = setup_environment()
        
        # Initialize tokenizer
        print("\n🔤 Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully!")
        
        # Create model configuration
        print("\n🏗️ Creating enhanced model configuration...")
        config = get_default_config()
        
        print("📊 Model Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Create model
        print("\n🤖 Creating enhanced model...")
        model = EnhancedMiniGPT(config).to(device)
        
        total_params, trainable_params = model.count_parameters()
        print(f"📈 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB")
        
        # Setup data manager with enhanced datasets
        print("\n📚 Setting up enhanced data manager...")
        data_manager = DataManager(
            tokenizer=tokenizer,
            max_length=config['max_seq_len'],
            batch_size=4
        )
        
        # Load high-quality training data
        print("\n🎯 Loading enhanced training datasets...")
        texts = data_manager.get_high_quality_datasets(sample_size=10000)
        
        if not texts:
            print("❌ Failed to load training data")
            return
        
        # Create dataloader
        train_dataloader = data_manager.create_dataloader(texts, shuffle=True)
        print(f"✅ Created training dataloader with {len(train_dataloader)} batches")
        
        # Setup trainer
        print("\n🎓 Setting up enhanced trainer...")
        training_config = get_training_config()
        trainer = EnhancedTrainer(model, tokenizer, device, training_config)
        
        # Train the model
        print("\n🚀 Starting enhanced training process...")
        training_losses = trainer.train(train_dataloader)
        
        # Save the trained model
        print("\n💾 Saving enhanced model...")
        model_path = trainer.save_final_model('enhanced_mini_chatgpt_v2.pt')
        
        # Plot training progress
        try:
            trainer.plot_training_progress()
        except Exception as e:
            print(f"⚠️ Could not plot training progress: {e}")
        
        # Setup text generator
        print("\n🎯 Setting up enhanced text generator...")
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
        
        print("\n🎉 Enhanced Mini-ChatGPT session completed!")
        print("✅ Model saved successfully!")
        print("🙏 Thank you for using Fopma-AI v2.0!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📦 Installing missing dependencies...")
        install_dependencies()
        print("🔄 Please restart and run the script again.")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please check your environment and try again.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()