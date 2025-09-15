"""
Configuration management for Fopma-AI
"""


def get_default_config():
    """Get default configuration for the model"""
    return {
        'vocab_size': 50257,  # GPT-2 tokenizer size
        'd_model': 384,       # Model dimension
        'num_heads': 12,      # Number of attention heads
        'num_layers': 6,      # Number of transformer layers
        'd_ff': 1536,        # Feed-forward dimension (4 * d_model)
        'max_seq_len': 256,   # Maximum sequence length
        'dropout': 0.1        # Dropout rate
    }


def get_training_config():
    """Get enhanced training configuration for better retraining"""
    return {
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'betas': (0.9, 0.95),
        'num_epochs': 100,  # Enhanced: 100 epochs for much better training
        'warmup_ratio': 0.1,
        'gradient_clipping': 1.0,
        'save_steps': 500,
        'logging_steps': 10,
        'batch_size': 4
    }


def get_generation_config():
    """Get default text generation configuration"""
    return {
        'max_length': 100,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
        'do_sample': True
    }


def get_data_config():
    """Get enhanced data processing configuration for better training"""
    return {
        'sample_size': 50000,  # Enhanced: 5x more training data (was 10000)
        'max_length': 256,
        'min_text_length': 50,
        'quality_filter': True,
        'num_workers': 0
    }