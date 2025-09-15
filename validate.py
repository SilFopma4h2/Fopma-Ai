#!/usr/bin/env python3
"""
Validation script for Mini-ChatGPT Colab notebook
This script validates the notebook structure and key components
"""

import json
import sys
import re
from pathlib import Path

def validate_notebook(notebook_path):
    """Validate the notebook structure and content"""
    
    # Load notebook
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"❌ Error loading notebook: {e}")
        return False
    
    print(f"✅ Successfully loaded notebook: {notebook_path}")
    
    # Check basic structure
    if 'cells' not in notebook:
        print("❌ No 'cells' found in notebook")
        return False
    
    cells = notebook['cells']
    print(f"✅ Found {len(cells)} cells")
    
    # Expected sections
    expected_sections = [
        "Install Required Packages",
        "Load and Stream The Pile Dataset", 
        "Data Processing and Tokenization",
        "Mini GPT Model Definition",
        "Training Configuration and Loop",
        "Chat Interface Implementation",
        "Interactive Chat Session",
        "Model Evaluation and Analysis",
        "Usage Instructions and Next Steps"
    ]
    
    # Check for key components
    notebook_text = ""
    for cell in cells:
        if cell.get('cell_type') == 'markdown':
            notebook_text += " ".join(cell.get('source', []))
        elif cell.get('cell_type') == 'code':
            notebook_text += " ".join(cell.get('source', []))
    
    # Check for required imports and classes
    required_components = [
        'transformers',
        'datasets', 
        'torch',
        'GPT2Tokenizer',
        'AdamW',
        'MultiHeadAttention',
        'TransformerBlock', 
        'MiniGPT',
        'TextDataset',
        'generate_text',
        'chat()',
        'load_dataset'
    ]
    
    missing_components = []
    for component in required_components:
        if component not in notebook_text:
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Missing required components: {missing_components}")
        return False
    else:
        print("✅ All required components found")
    
    # Check sections
    found_sections = []
    for section in expected_sections:
        if section in notebook_text:
            found_sections.append(section)
    
    print(f"✅ Found {len(found_sections)}/{len(expected_sections)} expected sections")
    
    # Check for proper markdown structure
    markdown_cells = [cell for cell in cells if cell.get('cell_type') == 'markdown']
    code_cells = [cell for cell in cells if cell.get('cell_type') == 'code']
    
    print(f"✅ Notebook structure: {len(markdown_cells)} markdown cells, {len(code_cells)} code cells")
    
    # Check for GPU/device handling
    if 'cuda' not in notebook_text or 'device' not in notebook_text:
        print("⚠️  GPU/device handling might be missing")
    else:
        print("✅ GPU/device handling found")
    
    # Check for proper model architecture
    architecture_keywords = ['attention', 'transformer', 'embedding', 'linear', 'layernorm']
    found_arch = [kw for kw in architecture_keywords if kw.lower() in notebook_text.lower()]
    print(f"✅ Model architecture components found: {found_arch}")
    
    # Check for training loop essentials
    training_keywords = ['optimizer', 'loss', 'backward', 'step', 'epoch', 'batch']
    found_training = [kw for kw in training_keywords if kw.lower() in notebook_text.lower()]
    print(f"✅ Training components found: {found_training}")
    
    print("\n🎉 Notebook validation completed successfully!")
    return True

def validate_readme(readme_path):
    """Validate README content"""
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
    except Exception as e:
        print(f"❌ Error loading README: {e}")
        return False
    
    print(f"✅ Successfully loaded README: {readme_path}")
    
    # Check for key sections
    required_sections = [
        '# 🤖 Fopma-AI',  # Updated to match new title
        'Mini-ChatGPT',
        'Features',
        'Quick Start', 
        'Model Architecture',
        'Training Configuration',
        'Interactive Usage Guide',  # Updated section name
        'Deployment & Production',  # Updated section name
        'Educational Deep Dive',    # New comprehensive section
        'License'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in readme_content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"❌ Missing README sections: {missing_sections}")
        return False
    else:
        print("✅ All required README sections found")
    
    return True

def main():
    """Main validation function"""
    print("🔍 Validating Mini-ChatGPT implementation...")
    print("=" * 50)
    
    # Check if files exist
    notebook_path = Path("mini_chatgpt_colab.ipynb")
    readme_path = Path("README.md")
    requirements_path = Path("requirements.txt")
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    if not readme_path.exists():
        print(f"❌ README not found: {readme_path}")
        return False
    
    if not requirements_path.exists():
        print(f"❌ Requirements file not found: {requirements_path}")
        return False
    
    # Validate notebook
    print("\n📓 Validating notebook...")
    print("-" * 30)
    if not validate_notebook(notebook_path):
        return False
    
    # Validate README
    print("\n📖 Validating README...")
    print("-" * 30)
    if not validate_readme(readme_path):
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All validations passed! The Mini-ChatGPT implementation is ready.")
    print("\n📋 Next steps:")
    print("1. Upload mini_chatgpt_colab.ipynb to Google Colab")
    print("2. Enable GPU runtime in Colab")
    print("3. Run all cells sequentially")
    print("4. Enjoy chatting with your mini-GPT!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)