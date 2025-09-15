# Fopma-Ai
My open-source AI project featuring a Mini-ChatGPT implementation

## 🚀 Mini-ChatGPT in Google Colab

This repository contains a complete implementation of a mini version of ChatGPT that can be trained and run in Google Colab. The implementation uses The Pile dataset from Hugging Face and creates a small but functional GPT model.

### 📋 Features

- **Dataset Streaming**: Efficiently streams The Pile dataset via Hugging Face datasets
- **GPT-2 Tokenization**: Uses the GPT-2 tokenizer for text processing
- **Mini GPT Architecture**: 
  - 2-4 Transformer layers
  - Hidden size: 128-256 dimensions
  - Same vocabulary size as GPT-2 tokenizer (~50,257 tokens)
- **Training Pipeline**:
  - AdamW optimizer with learning rate scheduling
  - Batch processing with gradient clipping
  - 1-2 epochs (optimized for Colab limitations)
- **Interactive Chat Interface**: Simple `chat()` function to interact with the trained model
- **Text Generation**: Configurable text generation with temperature, top-k, and top-p sampling

### 🛠️ Quick Start

1. **Open in Google Colab**:
   - Upload `mini_chatgpt_colab.ipynb` to Google Colab
   - Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)

2. **Run the Notebook**:
   - Execute all cells sequentially
   - The notebook will automatically install dependencies, load data, train the model, and provide a chat interface

3. **Interact with Your Model**:
   - Uncomment the `chat()` function call to start an interactive session
   - Type messages and get responses from your trained mini-GPT

### 📊 Model Architecture

```
MiniGPT(
  (token_embedding): Embedding(50257, 256)
  (position_embedding): Embedding(128, 256)
  (transformer_blocks): ModuleList(
    (0-3): 4 x TransformerBlock(
      (attention): MultiHeadAttention(8 heads, 256 dim)
      (feed_forward): FeedForward(256 → 1024 → 256)
      (norm1): LayerNorm(256)
      (norm2): LayerNorm(256)
    )
  )
  (ln_f): LayerNorm(256)
  (lm_head): Linear(256, 50257)
)
```

**Model Statistics**:
- Total Parameters: ~13.2M
- Trainable Parameters: ~13.2M
- Model Size: ~53 MB
- Maximum Sequence Length: 128 tokens

### 🎯 Training Configuration

- **Optimizer**: AdamW (lr=5e-4, weight_decay=0.01)
- **Scheduler**: Linear with warmup (100 warmup steps)
- **Batch Size**: 8 (optimized for Colab GPU memory)
- **Epochs**: 2 (can be adjusted based on available time)
- **Gradient Clipping**: Max norm of 1.0
- **Loss Function**: Cross-entropy with padding token ignoring

### 📈 Performance Expectations

This is an educational implementation optimized for Colab constraints. Expected behavior:
- **Training Time**: ~10-30 minutes (depending on dataset size and GPU)
- **Final Perplexity**: ~50-200 (varies based on data and training duration)
- **Response Quality**: Basic but coherent short responses
- **Memory Usage**: ~2-4 GB GPU memory

### 🔧 Customization Options

You can easily modify the model by adjusting these parameters in the notebook:

```python
model_config = {
    'vocab_size': len(tokenizer),
    'd_model': 256,          # Hidden size (128, 256, 512)
    'num_heads': 8,          # Attention heads (4, 8, 16)
    'num_layers': 4,         # Transformer layers (2, 4, 6, 8)
    'd_ff': 1024,           # Feed-forward dimension
    'max_seq_len': 128,     # Sequence length (64, 128, 256)
    'dropout': 0.1
}
```

**Training Parameters**:
```python
sample_size = 5000       # Number of training texts
batch_size = 8           # Batch size
num_epochs = 2           # Training epochs
learning_rate = 5e-4     # Learning rate
```

**Generation Parameters**:
```python
temperature = 0.8        # Randomness (0.1-2.0)
top_k = 50              # Top-k sampling
top_p = 0.95            # Nucleus sampling
max_length = 50         # Response length
```

### 📝 Example Usage

```python
# Generate text
response = generate_text(
    model, 
    tokenizer, 
    "The future of AI is", 
    max_length=50,
    temperature=0.8
)
print(response)
```

```python
# Start interactive chat
chat()
# Type your messages and get AI responses
# Type 'quit' to exit
```

### 🚧 Limitations

- **Model Size**: Small model with limited capabilities
- **Training Data**: Limited subset due to Colab constraints
- **Response Quality**: Basic responses, not production-ready
- **No Safety Filtering**: No content moderation or bias mitigation
- **Memory Constraints**: Limited by Colab's GPU memory
- **No Fine-tuning**: No instruction following or RLHF

### 🔬 Educational Value

This implementation demonstrates:
- **Transformer Architecture**: Complete implementation of attention mechanism
- **Language Model Training**: End-to-end training pipeline
- **Text Generation**: Various sampling strategies
- **Dataset Handling**: Streaming large datasets efficiently
- **PyTorch Best Practices**: Proper model design and training loops

### 🚀 Next Steps

To improve the model, consider:

1. **Scale Up**: Use larger models with more parameters
2. **More Data**: Train on larger datasets for longer periods
3. **Advanced Techniques**: Implement gradient accumulation, mixed precision
4. **Instruction Tuning**: Add instruction-following capabilities
5. **Safety**: Implement content filtering and bias mitigation
6. **Evaluation**: Add comprehensive evaluation metrics
7. **Deployment**: Create web interface or API

### 📚 Dependencies

The notebook automatically installs:
- `transformers` - Hugging Face transformers library
- `datasets` - Hugging Face datasets library
- `torch` - PyTorch deep learning framework
- `accelerate` - Hugging Face acceleration library
- `tqdm` - Progress bars
- `numpy` - Numerical computing

### 🤝 Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### ⚠️ Disclaimer

This is an educational implementation for learning purposes. The model is not suitable for production use and may generate inappropriate or biased content. Use responsibly and implement proper safety measures for any real-world applications.
