# 🚀 Fopma-AI Usage Examples

This document provides comprehensive examples of how to use the enhanced Fopma-AI Mini-ChatGPT implementation.

## 🎯 Quick Start Examples

### 🔥 Google Colab (Recommended)

**One-Command Setup:**
```bash
# 1. Clone the repository
!git clone https://github.com/SilFopma4h2/Fopma-Ai.git

# 2. Navigate to directory
%cd Fopma-Ai

# 3. Run the enhanced AI
!python main.py
```

**Manual Step-by-Step:**
```bash
# 1. Clone and setup
!git clone https://github.com/SilFopma4h2/Fopma-Ai.git
%cd Fopma-Ai

# 2. Validate setup (optional)
!python setup.py

# 3. Run main script
!python main.py

# Alternative: Use Jupyter notebook
# Open mini_chatgpt_colab.ipynb and run all cells
```

### 🖥️ Local Machine

```bash
# Clone repository
git clone https://github.com/SilFopma4h2/Fopma-Ai.git
cd Fopma-Ai

# Install dependencies
pip install -r requirements.txt

# Run the AI
python main.py
```

## 💬 Interactive Chat Examples

Once the model is trained, you'll see an interactive chat interface:

```
🤖 Enhanced Mini-ChatGPT Chat Interface
==================================================
Type your messages below. Type 'quit' to exit.
Commands:
  /temp <value>  - Set temperature (0.1-2.0)
  /length <num>  - Set max response length
  /reset         - Reset conversation
==================================================

🧑 You: Hello! How are you today?
🤖 AI: Hello! I'm doing well, thank you for asking. I'm excited to chat with you today. How can I help you?

🧑 You: Can you explain what machine learning is?
🤖 AI: Machine learning is a subset of artificial intelligence where algorithms learn patterns from data to make predictions or decisions without being explicitly programmed for each task.

🧑 You: /temp 1.5
🌡️ Temperature set to 1.5

🧑 You: Tell me a creative story
🤖 AI: Once upon a time, in a world where clouds were made of cotton candy and rainbows served as bridges between floating islands, there lived a curious little dragon named Sparkle who collected forgotten dreams...

🧑 You: /reset
🔄 Conversation reset

🧑 You: quit
👋 Goodbye!
```

## ⚙️ Configuration Examples

### 🎯 Model Size Configurations

**Small Model (for limited GPU memory):**
```python
# Edit in main.py
config = {
    'd_model': 256,        # Smaller dimension
    'num_heads': 8,        # Fewer heads
    'num_layers': 4,       # Fewer layers
    'max_seq_len': 128,    # Shorter sequences
    'dropout': 0.1
}
```

**Large Model (for powerful GPUs):**
```python
# Edit in main.py
config = {
    'd_model': 512,        # Larger dimension
    'num_heads': 16,       # More heads
    'num_layers': 8,       # More layers
    'max_seq_len': 512,    # Longer sequences  
    'dropout': 0.1
}
```

### 🎨 Generation Style Examples

**Conservative/Factual (Low Temperature):**
```python
# In interactive chat:
/temp 0.3
/length 100

# Result: More focused, factual responses
```

**Creative/Imaginative (High Temperature):**
```python
# In interactive chat:
/temp 1.5
/length 150

# Result: More creative, diverse responses
```

**Balanced (Default):**
```python
# In interactive chat:
/temp 0.8
/length 100

# Result: Good balance of creativity and coherence
```

## 🔧 Training Customization Examples

### 📊 Different Dataset Sizes

**Quick Testing (Fast training):**
```python
# Edit in main.py
sample_size = 1000      # Small dataset
num_epochs = 1          # Quick training
batch_size = 2          # Lower memory usage
```

**Full Training (Better quality):**
```python
# Edit in main.py
sample_size = 20000     # Larger dataset
num_epochs = 5          # More thorough training
batch_size = 8          # Higher throughput
```

### 🎯 Learning Rate Strategies

**Conservative (Stable but slow):**
```python
learning_rate = 1e-4    # Very low learning rate
```

**Aggressive (Fast but risky):**
```python
learning_rate = 5e-3    # Higher learning rate
```

**Adaptive (Recommended):**
```python
# Uses warmup + linear decay (default in main.py)
warmup_steps = int(0.1 * total_steps)
```

## 🌐 Deployment Examples

### 🎨 Gradio Web Interface

```python
# Create gradio_demo.py
import gradio as gr
import torch
from main import enhanced_mini_gpt, enhanced_text_generation

# Load trained model
model, tokenizer, config, device = enhanced_mini_gpt()
checkpoint = torch.load('enhanced_mini_chatgpt.pt')
model.load_state_dict(checkpoint['model_state_dict'])

def chat_interface(message, history, temperature, max_length):
    """Chat interface for Gradio"""
    response = generate_text(
        message, 
        max_length=max_length, 
        temperature=temperature
    )
    history.append((message, response))
    return history, ""

# Create Gradio interface
with gr.Blocks(title="Fopma-AI ChatBot") as app:
    gr.Markdown("# 🤖 Fopma-AI Enhanced Mini-ChatGPT")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(
                placeholder="Type your message here...",
                label="Your Message"
            )
            
        with gr.Column(scale=1):
            temperature = gr.Slider(
                0.1, 2.0, value=0.8, 
                label="Temperature (Creativity)"
            )
            max_length = gr.Slider(
                10, 200, value=100,
                label="Response Length"
            )
            clear = gr.Button("Clear Chat")
    
    msg.submit(
        chat_interface, 
        [msg, chatbot, temperature, max_length], 
        [chatbot, msg]
    )
    clear.click(lambda: [], None, chatbot)

# Launch app
app.launch(share=True)
```

### ⚡ FastAPI REST API

```python
# Create api_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from main import enhanced_mini_gpt

app = FastAPI(title="Fopma-AI API")

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model, tokenizer, device
    model, tokenizer, config, device = enhanced_mini_gpt()
    checkpoint = torch.load('enhanced_mini_chatgpt.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.8
    max_length: int = 100

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = generate_text(
        request.message,
        max_length=request.max_length,
        temperature=request.temperature
    )
    return {"response": response}

# Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### 📱 Streamlit Dashboard

```python
# Create streamlit_app.py
import streamlit as st
import torch
from main import enhanced_mini_gpt

st.set_page_config(page_title="Fopma-AI Chat", page_icon="🤖")

# Load model (cached)
@st.cache_resource
def load_model():
    model, tokenizer, config, device = enhanced_mini_gpt()
    checkpoint = torch.load('enhanced_mini_chatgpt.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, tokenizer, device

model, tokenizer, device = load_model()

st.title("🤖 Fopma-AI Enhanced Mini-ChatGPT")

# Sidebar controls
st.sidebar.header("🎛️ Generation Settings")
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8)
max_length = st.sidebar.slider("Max Length", 10, 200, 100)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_text(
                prompt, 
                max_length=max_length, 
                temperature=temperature
            )
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Run with: streamlit run streamlit_app.py
```

## 🧪 Advanced Usage Examples

### 🔬 Custom Dataset Training

```python
# Custom dataset example
def train_on_custom_data():
    """Train model on your own text data"""
    
    # Load your custom texts
    with open('my_dataset.txt', 'r') as f:
        custom_texts = f.readlines()
    
    # Filter and clean
    custom_texts = [
        text.strip() for text in custom_texts 
        if len(text.strip()) > 50  # Remove short texts
    ]
    
    # Replace the dataset loading in main.py with:
    # texts = custom_texts
    
    print(f"Training on {len(custom_texts)} custom texts")
```

### 📊 Model Evaluation

```python
# Evaluation example
def evaluate_model(model, tokenizer, device):
    """Evaluate model performance"""
    
    test_prompts = [
        "The weather today is",
        "Machine learning is",
        "In the future, AI will",
        "The most important thing in life is"
    ]
    
    print("🔍 Model Evaluation:")
    print("=" * 40)
    
    for prompt in test_prompts:
        response = generate_text(
            prompt, 
            max_length=50, 
            temperature=0.8
        )
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 40)
```

### 🎯 Fine-tuning for Specific Tasks

```python
# Task-specific fine-tuning example
def create_qa_bot():
    """Fine-tune for question-answering"""
    
    # Prepare Q&A dataset
    qa_pairs = [
        "Q: What is Python? A: Python is a programming language.",
        "Q: How does AI work? A: AI learns patterns from data.",
        # Add more Q&A pairs...
    ]
    
    # Train on Q&A format
    # Replace dataset in main.py with qa_pairs
    print("Training Q&A bot...")

def create_story_writer():
    """Fine-tune for creative writing"""
    
    # Prepare story dataset
    stories = [
        "Once upon a time, in a magical kingdom...",
        "The spaceship landed on the alien planet...",
        # Add more creative texts...
    ]
    
    # Train on creative content
    print("Training creative writer...")
```

## 🐛 Troubleshooting Examples

### 💾 Memory Issues

```python
# If you get CUDA out of memory:

# Solution 1: Reduce batch size
batch_size = 2  # Instead of 4

# Solution 2: Reduce model size
config['d_model'] = 256  # Instead of 384
config['num_layers'] = 4  # Instead of 6

# Solution 3: Clear GPU cache
import torch
torch.cuda.empty_cache()
```

### 🐌 Slow Training

```python
# Speed optimization:

# Use mixed precision (if supported)
from torch.cuda.amp import autocast, GradScaler

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Reduce dataset size for testing
sample_size = 1000  # Instead of 10000
```

### 🎭 Poor Response Quality

```python
# Quality improvements:

# Lower temperature for more focused responses
temperature = 0.3  # Instead of 0.8

# Train for more epochs
num_epochs = 5  # Instead of 3

# Use larger model (if memory allows)
config['d_model'] = 512
config['num_layers'] = 8
```

## 🎓 Educational Examples

### 📚 Understanding Attention

```python
# Visualize attention patterns (advanced)
def visualize_attention():
    """Example of attention visualization"""
    
    text = "The cat sat on the mat"
    tokens = tokenizer.encode(text)
    
    with torch.no_grad():
        outputs = model(torch.tensor([tokens]))
        # Extract attention weights from model
        # attention_weights = outputs.attentions
    
    print("Attention visualization would show which words")
    print("the model focuses on when predicting each token.")
```

### 🔬 Experiment with Hyperparameters

```python
# Hyperparameter experiments
experiments = [
    {'lr': 1e-4, 'batch_size': 2, 'name': 'conservative'},
    {'lr': 3e-4, 'batch_size': 4, 'name': 'balanced'},
    {'lr': 5e-4, 'batch_size': 8, 'name': 'aggressive'},
]

for exp in experiments:
    print(f"Running experiment: {exp['name']}")
    # Modify training parameters and run
    # Compare results
```

---

## 🎉 Getting Started

Ready to try these examples? Start with the Quick Start section and work your way through the examples that interest you most!

For more help, check the main [README.md](README.md) file or open an issue on GitHub.

**Happy AI building!** 🚀🤖