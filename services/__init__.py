"""
Servicios de la aplicación CARID ALPR
"""

from .file_service import file_service, FileService
from .detection_service import detection_service, DetectionService

__all__ = [
    "file_service",
    "FileService",
    "detection_service",
    "DetectionService"
]