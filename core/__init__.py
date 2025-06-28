# core/__init__.py
"""
Funcionalidades centrales para CARID ALPR
"""

from . import utils
from .plate_filters import PlateValidator
from .roi_processor import ROIProcessor
from .enhanced_pipeline import EnhancedALPRPipeline


__all__ = [
    "utils",
    "PlateValidator",
    "ROIProcessor",
    "EnhancedALPRPipeline"
]