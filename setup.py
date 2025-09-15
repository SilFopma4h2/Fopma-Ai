#!/usr/bin/env python3
"""
Fopma-AI Setup Script
Quick setup and validation for the enhanced Mini-ChatGPT implementation

Usage in Google Colab:
!git clone https://github.com/SilFopma4h2/Fopma-Ai.git
%cd Fopma-Ai
!python setup.py
"""

import os
import sys
import subprocess
import importlib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def print_banner():
    """Print welcome banner"""
    print("🚀" + "=" * 60 + "🚀")
    print("🤖  FOPMA-AI ENHANCED MINI-CHATGPT SETUP  🤖")
    print("🚀" + "=" * 60 + "🚀")
    print("🌟 Setting up your enhanced AI experience...")
    print("✨ This will take 2-3 minutes on first run")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_gpu():
    """Check GPU availability"""
    print("\n🔥 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   Memory: {memory_gb:.1f} GB")
            
            if memory_gb < 4:
                print("⚠️  Warning: Low GPU memory. Consider reducing model size.")
            elif memory_gb >= 15:
                print("🚀 Excellent! High-end GPU detected. You can use larger models.")
            else:
                print("👍 Good GPU memory for standard training.")
                
            return True
        else:
            print("⚠️  No GPU detected. Training will be slow on CPU.")
            print("   Consider enabling GPU in Colab: Runtime → Change runtime type → GPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet - will install shortly")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Core packages that are usually needed
    core_packages = [
        "torch>=1.9.0",
        "transformers>=4.20.0", 
        "datasets>=2.0.0",
        "accelerate>=0.20.0",
        "tqdm>=4.60.0",
        "numpy>=1.21.0"
    ]
    
    print("   Installing core packages...")
    for package in core_packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", package
            ])
            print(f"   ✅ {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  Failed to install {package}")
    
    print("   ✅ Core dependencies installed!")

def validate_installation():
    """Validate that everything is working"""
    print("\n🔍 Validating installation...")
    
    required_modules = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("numpy", "NumPy"),
        ("tqdm", "TQDM")
    ]
    
    all_good = True
    for module_name, display_name in required_modules:
        try:
            importlib.import_module(module_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            print(f"   ❌ {display_name} - Installation failed")
            all_good = False
    
    return all_good

def check_files():
    """Check that all required files are present"""
    print("\n📁 Checking project files...")
    
    required_files = [
        ("main.py", "Main script"),
        ("mini_chatgpt_colab.ipynb", "Jupyter notebook"),
        ("requirements.txt", "Requirements file"),
        ("validate.py", "Validation script"),
        ("README.md", "Documentation")
    ]
    
    all_present = True
    for file_name, description in required_files:
        if os.path.exists(file_name):
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description} - Missing: {file_name}")
            all_present = False
    
    return all_present

def run_quick_test():
    """Run a quick test to ensure everything works"""
    print("\n🧪 Running quick functionality test...")
    
    try:
        # Test basic imports
        import torch
        import transformers
        from transformers import GPT2Tokenizer
        
        print("   ✅ Core imports successful")
        
        # Test tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        
        print("   ✅ Tokenizer working")
        
        # Test basic tensor operations
        if torch.cuda.is_available():
            device = torch.device('cuda')
            test_tensor = torch.randn(2, 3).to(device)
            print("   ✅ GPU operations working")
        else:
            test_tensor = torch.randn(2, 3)
            print("   ✅ CPU operations working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "🎯" + "=" * 50 + "🎯")
    print("🎉 SETUP COMPLETE! Next steps:")
    print("🎯" + "=" * 50 + "🎯")
    print()
    print("🚀 Quick Start:")
    print("   python main.py")
    print()
    print("📔 Or use Jupyter notebook:")
    print("   Open mini_chatgpt_colab.ipynb and run all cells")
    print()
    print("🔧 Advanced options:")
    print("   Edit main.py to customize model size and training")
    print()
    print("📚 Need help?")
    print("   Check README.md for detailed instructions")
    print("   View troubleshooting section for common issues")
    print()
    print("🌟 Enjoy your enhanced AI experience!")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    # Run all checks and setup steps
    steps = [
        ("Python version", check_python_version),
        ("GPU availability", check_gpu), 
        ("Dependencies", install_dependencies),
        ("Installation validation", validate_installation),
        ("Project files", check_files),
        ("Quick functionality test", run_quick_test)
    ]
    
    success = True
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            success = False
            break
    
    if success:
        print("\n✅ All setup steps completed successfully!")
        print_next_steps()
        return 0
    else:
        print("\n❌ Setup encountered issues. Please check the errors above.")
        print("💡 Try running this script again or check README.md for troubleshooting.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)