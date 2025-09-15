#!/usr/bin/env python3
"""
Enhanced test script for Fopma-AI modular architecture
Tests the new file structure and training data improvements
"""

import sys
import os
import tempfile

def test_modular_structure():
    """Test that the new modular structure is working"""
    print("🔍 Testing modular structure...")
    
    # Test directory structure
    required_directories = [
        "fopma_ai",
        "fopma_ai/models", 
        "fopma_ai/training",
        "fopma_ai/data",
        "fopma_ai/generation",
        "fopma_ai/utils"
    ]
    
    missing_dirs = []
    for directory in required_directories:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ All required directories present")
    
    # Test module files
    required_files = [
        "fopma_ai/__init__.py",
        "fopma_ai/models/__init__.py",
        "fopma_ai/models/attention.py",
        "fopma_ai/models/transformer.py", 
        "fopma_ai/models/mini_gpt.py",
        "fopma_ai/training/__init__.py",
        "fopma_ai/training/trainer.py",
        "fopma_ai/data/__init__.py",
        "fopma_ai/data/dataset.py",
        "fopma_ai/data/data_manager.py",
        "fopma_ai/generation/__init__.py",
        "fopma_ai/generation/generator.py",
        "fopma_ai/utils/__init__.py",
        "fopma_ai/utils/environment.py",
        "fopma_ai/utils/config.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required module files present")
    return True

def test_module_imports():
    """Test that all modules can be imported successfully"""
    print("🔍 Testing module imports...")
    
    try:
        # Test main package import
        import fopma_ai
        print("✅ Main package imported")
        
        # Test individual modules
        from fopma_ai.models import EnhancedMiniGPT
        print("✅ Models module imported")
        
        from fopma_ai.training import EnhancedTrainer
        print("✅ Training module imported")
        
        from fopma_ai.data import DataManager
        print("✅ Data module imported")
        
        from fopma_ai.generation import TextGenerator
        print("✅ Generation module imported")
        
        from fopma_ai.utils import setup_environment, install_dependencies
        print("✅ Utils module imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation and basic functionality"""
    print("🔍 Testing model creation...")
    
    try:
        from fopma_ai.models import EnhancedMiniGPT
        from fopma_ai.utils.config import get_default_config
        import torch
        
        # Create configuration
        config = get_default_config()
        print("✅ Configuration created")
        
        # Create model
        model = EnhancedMiniGPT(config)
        print("✅ Model created successfully")
        
        # Test parameter counting
        total_params, trainable_params = model.count_parameters()
        print(f"✅ Model has {total_params:,} total parameters")
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        
        with torch.no_grad():
            logits = model(input_ids)
        
        expected_shape = (batch_size, seq_len, config['vocab_size'])
        if logits.shape == expected_shape:
            print("✅ Model forward pass works correctly")
            return True
        else:
            print(f"❌ Unexpected output shape: {logits.shape}, expected: {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_data_manager():
    """Test enhanced data management functionality"""
    print("🔍 Testing data manager...")
    
    try:
        from fopma_ai.data import DataManager
        from transformers import GPT2Tokenizer
        
        # Create tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer created")
        
        # Create data manager
        data_manager = DataManager(tokenizer, max_length=128, batch_size=2)
        print("✅ Data manager created")
        
        # Test fallback data generation
        texts = data_manager._get_fallback_texts()
        if len(texts) > 0:
            print(f"✅ Fallback texts generated: {len(texts)} texts")
        else:
            print("❌ No fallback texts generated")
            return False
        
        # Test dataloader creation
        dataloader = data_manager.create_dataloader(texts[:10], shuffle=True)
        print(f"✅ DataLoader created with {len(dataloader)} batches")
        
        # Test data quality filtering
        from fopma_ai.data.dataset import ImprovedTextDataset
        dataset = ImprovedTextDataset(texts[:5], tokenizer, 128, quality_filter=True)
        print(f"✅ Quality filtering works: {len(dataset)} texts after filtering")
        
        return True
        
    except Exception as e:
        print(f"❌ Data manager error: {e}")
        return False

def test_training_components():
    """Test training components"""
    print("🔍 Testing training components...")
    
    try:
        from fopma_ai.training import EnhancedTrainer
        from fopma_ai.models import EnhancedMiniGPT
        from fopma_ai.utils.config import get_default_config, get_training_config
        from transformers import GPT2Tokenizer
        import torch
        
        # Create components
        config = get_default_config()
        model = EnhancedMiniGPT(config)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = torch.device('cpu')  # Use CPU for testing
        
        # Create trainer
        training_config = get_training_config()
        trainer = EnhancedTrainer(model, tokenizer, device, training_config)
        print("✅ Trainer created successfully")
        
        # Test configuration methods
        default_config = trainer._get_default_config()
        if isinstance(default_config, dict):
            print("✅ Default training config is valid")
        else:
            print("❌ Invalid default training config")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Training components error: {e}")
        return False

def test_generation_components():
    """Test text generation components"""
    print("🔍 Testing generation components...")
    
    try:
        from fopma_ai.generation import TextGenerator
        from fopma_ai.models import EnhancedMiniGPT
        from fopma_ai.utils.config import get_default_config
        from transformers import GPT2Tokenizer
        import torch
        
        # Create components
        config = get_default_config()
        model = EnhancedMiniGPT(config)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = torch.device('cpu')
        
        # Create generator
        generator = TextGenerator(model, tokenizer, device)
        print("✅ Text generator created successfully")
        
        # Test text generation (short generation for testing)
        prompt = "The future is"
        generated_text = generator.generate_text(prompt, max_length=10, temperature=1.0)
        
        if generated_text and len(generated_text) > len(prompt):
            print("✅ Text generation works")
        else:
            print("❌ Text generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Generation components error: {e}")
        return False

def test_backward_compatibility():
    """Test that the modular main script works"""
    print("🔍 Testing backward compatibility...")
    
    try:
        # Check that main_modular.py exists and compiles
        if not os.path.exists("main_modular.py"):
            print("❌ main_modular.py not found")
            return False
        
        # Test compilation
        import py_compile
        py_compile.compile("main_modular.py", doraise=True)
        print("✅ main_modular.py compiles successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀" + "="*50 + "🚀")
    print("🧪 FOPMA-AI MODULAR ARCHITECTURE TESTS")
    print("🚀" + "="*50 + "🚀")
    print()
    
    tests = [
        ("Modular Structure", test_modular_structure),
        ("Module Imports", test_module_imports),
        ("Model Creation", test_model_creation),
        ("Data Manager", test_data_manager),
        ("Training Components", test_training_components),
        ("Generation Components", test_generation_components),
        ("Backward Compatibility", test_backward_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "="*60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Modular architecture is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    print("\n✅ Enhanced Fopma-AI features:")
    print("   🏗️ Modular file structure for better maintainability")
    print("   📚 Multiple high-quality dataset sources with quality filtering")
    print("   🚀 Enhanced training with better optimization strategies")
    print("   🎯 Advanced text generation with multiple sampling methods")
    print("   💬 Improved interactive chat interface")
    print("   🔧 Automatic system optimization and configuration")

if __name__ == "__main__":
    main()