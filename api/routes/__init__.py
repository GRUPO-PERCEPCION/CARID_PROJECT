"""
Rutas de la API CARID ALPR
"""

from . import health
from . import detection
from . import video

__all__ = [
    "health",
    "detection",
    "video"
]