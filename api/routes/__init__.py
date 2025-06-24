# api/routes/__init__.py
"""
Rutas de la API CARID ALPR
"""

from . import health
from . import detection

__all__ = [
    "health",
    "detection"
]