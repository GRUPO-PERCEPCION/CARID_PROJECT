"""
Esquemas de datos para la API CARID ALPR
"""

from .detection import (
    # Esquemas básicos
    BoundingBox,
    CharacterDetection,
    PlateDetection,

    # Resultados
    CharacterRecognitionResult,
    PlateDetectionResult,
    ProcessedPlate,
    ImageProcessingResult,

    # Requests
    DetectionRequest,
    ImageUploadInfo,

    # Responses
    ImageDetectionResponse,
    SuccessResponse,
    ErrorResponse,
    ErrorDetail,
    ValidationError,
    HealthCheckResponse
)

__all__ = [
    # Esquemas básicos
    "BoundingBox",
    "CharacterDetection",
    "PlateDetection",

    # Resultados
    "CharacterRecognitionResult",
    "PlateDetectionResult",
    "ProcessedPlate",
    "ImageProcessingResult",

    # Requests
    "DetectionRequest",
    "ImageUploadInfo",

    # Responses
    "ImageDetectionResponse",
    "SuccessResponse",
    "ErrorResponse",
    "ErrorDetail",
    "ValidationError",
    "HealthCheckResponse"
]