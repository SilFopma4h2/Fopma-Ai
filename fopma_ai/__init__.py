"""
Fopma-AI: Enhanced Mini-ChatGPT Implementation

A modular, production-ready Mini-ChatGPT implementation designed for educational
and research purposes with enterprise-level architecture improvements.
"""

__version__ = "2.0.0"
__author__ = "Fopma-AI Team"
__license__ = "MIT"

from .models import EnhancedMiniGPT
from .training import EnhancedTrainer
from .data import DataManager
from .generation import TextGenerator
from .utils import setup_environment, install_dependencies

__all__ = [
    "EnhancedMiniGPT",
    "EnhancedTrainer", 
    "DataManager",
    "TextGenerator",
    "setup_environment",
    "install_dependencies"
]