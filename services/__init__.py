"""
Servicios de la aplicación CARID ALPR
"""

from .file_service import file_service, FileService
from .detection_service import detection_service, DetectionService
from .video_service import video_service, VideoService

__all__ = [
    "file_service",
    "FileService",
    "detection_service",
    "DetectionService",
    "video_service",
    "VideoService"
]