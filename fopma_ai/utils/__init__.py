"""
Utility functions for Fopma-AI
"""

from .environment import setup_environment, install_dependencies
from .config import get_default_config

__all__ = [
    "setup_environment",
    "install_dependencies", 
    "get_default_config"
]