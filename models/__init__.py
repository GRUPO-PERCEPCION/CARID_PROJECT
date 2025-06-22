# models/__init__.py
"""
Módulo de modelos para CARID ALPR
"""

from .model_manager import model_manager
from .base_model import BaseModel

__all__ = ["model_manager", "BaseModel"]
