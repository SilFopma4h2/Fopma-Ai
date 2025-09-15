"""
Environment setup and dependency management
"""

import sys
import os
import warnings
import torch
import numpy as np
import random

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
    print("🌟 Setting up environment...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable optimizations for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("   Running on CPU - consider using GPU for better performance")
    
    return device


def check_system_requirements():
    """Check if system meets minimum requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("⚠️ Warning: Python 3.7+ recommended")
    else:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}")
    
    # Check PyTorch installation
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
        else:
            print("⚠️ CUDA not available - will use CPU")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check available memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 4:
            print("⚠️ Warning: GPU memory < 4GB - may need to reduce batch size")
        else:
            print(f"✅ GPU memory: {gpu_memory:.1f} GB")
    
    return True


def get_model_config():
    """Get optimized model configuration based on available resources"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        # GPU configuration
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory >= 8:
            # High-end GPU
            config = {
                'vocab_size': 50257,
                'd_model': 512,
                'num_heads': 16,
                'num_layers': 8,
                'd_ff': 2048,
                'max_seq_len': 512,
                'dropout': 0.1,
                'batch_size': 8
            }
        elif gpu_memory >= 4:
            # Mid-range GPU
            config = {
                'vocab_size': 50257,
                'd_model': 384,
                'num_heads': 12,
                'num_layers': 6,
                'd_ff': 1536,
                'max_seq_len': 256,
                'dropout': 0.1,
                'batch_size': 4
            }
        else:
            # Low-end GPU
            config = {
                'vocab_size': 50257,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 4,
                'd_ff': 1024,
                'max_seq_len': 128,
                'dropout': 0.1,
                'batch_size': 2
            }
    else:
        # CPU configuration (smaller model)
        config = {
            'vocab_size': 50257,
            'd_model': 192,
            'num_heads': 6,
            'num_layers': 3,
            'd_ff': 768,
            'max_seq_len': 128,
            'dropout': 0.1,
            'batch_size': 1
        }
    
    print(f"📊 Auto-configured for {device.type.upper()}:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    return config