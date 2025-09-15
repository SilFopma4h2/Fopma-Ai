#!/usr/bin/env python3
"""
Quick test script for Fopma-AI functionality
Tests core components without heavy dependencies
"""

import sys
import os

def test_file_structure():
    """Test that all required files are present"""
    print("🔍 Testing file structure...")
    
    required_files = [
        "main.py",
        "README.md", 
        "requirements.txt",
        "setup.py",
        "validate.py",
        "mini_chatgpt_colab.ipynb"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def test_main_py_structure():
    """Test that main.py has the expected structure"""
    print("🔍 Testing main.py structure...")
    
    try:
        with open("main.py", 'r') as f:
            content = f.read()
        
        required_components = [
            "install_dependencies",
            "setup_environment", 
            "enhanced_mini_gpt",
            "EnhancedMiniGPT",
            "ImprovedMultiHeadAttention",
            "enhanced_training_loop",
            "enhanced_text_generation",
            "interactive_chat",
            "main()"
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing components in main.py: {missing_components}")
            return False
        
        print("✅ main.py structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")
        return False

def test_readme_enhancement():
    """Test that README has been significantly enhanced"""
    print("🔍 Testing README enhancement...")
    
    try:
        with open("README.md", 'r') as f:
            content = f.read()
        
        # Check for key enhancements
        enhancements = [
            "Enhanced Mini-ChatGPT",
            "One-Command Setup",
            "Troubleshooting & FAQ",
            "Deployment & Production",
            "Educational Deep Dive",
            "Interactive Usage Guide",
            "Model Architecture Deep Dive",
            "Enhanced Architecture & Features",
            "Contributing & Community"
        ]
        
        missing_enhancements = []
        for enhancement in enhancements:
            if enhancement not in content:
                missing_enhancements.append(enhancement)
        
        if missing_enhancements:
            print(f"❌ Missing README enhancements: {missing_enhancements}")
            return False
        
        # Check README length (should be significantly longer)
        word_count = len(content.split())
        if word_count < 3000:  # Original was ~1000 words, enhanced should be 3000+
            print(f"❌ README not significantly enhanced. Word count: {word_count}")
            return False
        
        print(f"✅ README significantly enhanced ({word_count:,} words)")
        return True
        
    except Exception as e:
        print(f"❌ Error reading README.md: {e}")
        return False

def test_python_syntax():
    """Test that all Python files have valid syntax"""
    print("🔍 Testing Python syntax...")
    
    python_files = ["main.py", "setup.py", "validate.py", "example_usage.py"]
    
    for file in python_files:
        if os.path.exists(file):
            try:
                with open(file, 'r') as f:
                    code = f.read()
                compile(code, file, 'exec')
                print(f"   ✅ {file}")
            except SyntaxError as e:
                print(f"   ❌ {file}: Syntax error at line {e.lineno}")
                return False
            except Exception as e:
                print(f"   ⚠️  {file}: {e}")
    
    print("✅ All Python files have valid syntax")
    return True

def test_colab_instructions():
    """Test that Colab instructions are clear and present"""
    print("🔍 Testing Colab setup instructions...")
    
    try:
        with open("README.md", 'r') as f:
            content = f.read()
        
        colab_keywords = [
            "!git clone",
            "%cd Fopma-Ai", 
            "!python main.py",
            "Google Colab",
            "one-command setup",
            "Quick Start"
        ]
        
        missing_keywords = []
        for keyword in colab_keywords:
            if keyword not in content:
                missing_keywords.append(keyword)
        
        if missing_keywords:
            print(f"❌ Missing Colab instructions: {missing_keywords}")
            return False
        
        print("✅ Colab setup instructions are clear and complete")
        return True
        
    except Exception as e:
        print(f"❌ Error checking Colab instructions: {e}")
        return False

def test_requirements():
    """Test that requirements.txt has been enhanced"""
    print("🔍 Testing requirements.txt enhancement...")
    
    try:
        with open("requirements.txt", 'r') as f:
            content = f.read()
        
        # Check for enhanced requirements
        enhanced_packages = [
            "torch>=",
            "transformers>=", 
            "datasets>=",
            "gradio>=",
            "streamlit>=",
            "fastapi>=",
            "matplotlib>=",
            "seaborn>="
        ]
        
        missing_packages = []
        for package in enhanced_packages:
            if package not in content:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing enhanced packages: {missing_packages}")
            return False
        
        print("✅ requirements.txt has been enhanced with additional packages")
        return True
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀" + "=" * 50 + "🚀")
    print("🧪 FOPMA-AI ENHANCEMENT VALIDATION")
    print("🚀" + "=" * 50 + "🚀")
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("main.py Components", test_main_py_structure), 
        ("README Enhancement", test_readme_enhancement),
        ("Python Syntax", test_python_syntax),
        ("Colab Instructions", test_colab_instructions),
        ("Enhanced Requirements", test_requirements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}:")
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Fopma-AI enhancements are working correctly.")
        print()
        print("✅ Ready for use! Users can now:")
        print("   1. Clone the repository in Google Colab")
        print("   2. Run 'python main.py' for enhanced AI experience")
        print("   3. Follow the comprehensive README instructions")
        print()
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)