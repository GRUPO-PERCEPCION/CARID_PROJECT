from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime


class BoundingBox(BaseModel):
    """Esquema para bounding boxes"""
    x1: float = Field(..., description="Coordenada X superior izquierda")
    y1: float = Field(..., description="Coordenada Y superior izquierda")
    x2: float = Field(..., description="Coordenada X inferior derecha")
    y2: float = Field(..., description="Coordenada Y inferior derecha")

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class CharacterDetection(BaseModel):
    """Esquema para detección de caracteres individuales"""
    character: str = Field(..., description="Carácter reconocido")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza de la detección")
    class_id: int = Field(..., description="ID de la clase")
    area: float = Field(..., description="Área del bounding box")
    center_x: float = Field(..., description="Coordenada X del centro")
    center_y: float = Field(..., description="Coordenada Y del centro")
    width: float = Field(..., description="Ancho del carácter")
    height: float = Field(..., description="Alto del carácter")


class PlateDetection(BaseModel):
    """Esquema para detección de placas"""
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza de la detección")
    area: float = Field(..., description="Área del bounding box")
    aspect_ratio: float = Field(..., description="Relación de aspecto (ancho/alto)")
    width: float = Field(..., description="Ancho de la placa")
    height: float = Field(..., description="Alto de la placa")
    center: List[float] = Field(..., description="Centro [x, y]")
    normalized_bbox: List[float] = Field(..., description="Bbox normalizado [0-1]")


class CharacterRecognitionResult(BaseModel):
    """Resultado del reconocimiento de caracteres"""
    success: bool = Field(..., description="Si el reconocimiento fue exitoso")
    characters_detected: int = Field(..., description="Número de caracteres detectados")
    plate_text: str = Field(..., description="Texto completo de la placa")
    is_valid_format: bool = Field(..., description="Si el formato es válido para placas peruanas")
    characters: List[CharacterDetection] = Field(default=[], description="Lista de caracteres detectados")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza general")
    image_shape: Dict[str, int] = Field(..., description="Dimensiones de la imagen")
    processing_info: Dict[str, Any] = Field(..., description="Información de procesamiento")


class PlateDetectionResult(BaseModel):
    """Resultado de la detección de placas"""
    success: bool = Field(..., description="Si la detección fue exitosa")
    plates_detected: int = Field(..., description="Número de placas detectadas")
    plates: List[PlateDetection] = Field(default=[], description="Lista de placas detectadas")
    image_shape: Dict[str, int] = Field(..., description="Dimensiones de la imagen")
    processing_info: Dict[str, Any] = Field(..., description="Información de procesamiento")


class ProcessedPlate(BaseModel):
    """Esquema para una placa completamente procesada"""
    plate_id: int = Field(..., description="ID único de la placa")
    plate_bbox: List[float] = Field(..., description="Bounding box de la placa")
    plate_confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza de detección de placa")
    plate_area: float = Field(..., description="Área de la placa")
    character_recognition: CharacterRecognitionResult = Field(..., description="Resultado de reconocimiento")
    plate_text: str = Field(..., description="Texto final de la placa")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza combinada")
    is_valid_plate: bool = Field(..., description="Si es una placa válida")


class ImageProcessingResult(BaseModel):
    """Resultado completo del procesamiento de imagen"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Información básica
    success: bool = Field(..., description="Si el procesamiento fue exitoso")
    message: str = Field(default="", description="Mensaje informativo")

    # Información de archivo
    file_info: Dict[str, Any] = Field(..., description="Información del archivo procesado")

    # Resultados de procesamiento
    plates_processed: int = Field(..., description="Número de placas procesadas")
    plate_detection: Optional[PlateDetectionResult] = Field(None, description="Resultado de detección")
    final_results: List[ProcessedPlate] = Field(default=[], description="Resultados finales")
    best_result: Optional[ProcessedPlate] = Field(None, description="Mejor resultado")

    # Resumen
    processing_summary: Dict[str, int] = Field(..., description="Resumen de procesamiento")

    # Metadatos
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp del procesamiento")

    # URLs de resultados (si aplica)
    result_urls: Optional[Dict[str, str]] = Field(None, description="URLs de archivos de resultado")


class ImageUploadInfo(BaseModel):
    """Información de archivo subido"""
    filename: str = Field(..., description="Nombre del archivo")
    original_filename: str = Field(..., description="Nombre original del archivo")
    content_type: str = Field(..., description="Tipo de contenido")
    size_bytes: int = Field(..., description="Tamaño en bytes")
    size_mb: float = Field(..., description="Tamaño en MB")
    dimensions: Optional[Dict[str, int]] = Field(None, description="Dimensiones [width, height]")


class DetectionRequest(BaseModel):
    """Esquema de request para detección"""
    confidence_threshold: Optional[float] = Field(0.5, ge=0.1, le=1.0, description="Umbral de confianza")
    iou_threshold: Optional[float] = Field(0.4, ge=0.1, le=1.0, description="Umbral IoU")
    max_detections: Optional[int] = Field(5, ge=1, le=10, description="Máximo número de detecciones")
    enhance_image: Optional[bool] = Field(False, description="Aplicar mejoras a la imagen")
    return_visualization: Optional[bool] = Field(False, description="Retornar imagen con visualizaciones")
    save_results: Optional[bool] = Field(True, description="Guardar resultados")


# Esquemas de error
class ErrorDetail(BaseModel):
    """Detalle de error"""
    error_type: str = Field(..., description="Tipo de error")
    message: str = Field(..., description="Mensaje de error")
    details: Optional[Dict[str, Any]] = Field(None, description="Detalles adicionales")


class ValidationError(BaseModel):
    """Error de validación"""
    field: str = Field(..., description="Campo con error")
    message: str = Field(..., description="Mensaje de error")
    value: Any = Field(..., description="Valor que causó el error")


# Respuestas estándar
class SuccessResponse(BaseModel):
    """Respuesta exitosa estándar"""
    success: bool = Field(True, description="Operación exitosa")
    message: str = Field(..., description="Mensaje de éxito")
    data: Optional[Any] = Field(None, description="Datos de respuesta")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Respuesta de error estándar"""
    success: bool = Field(False, description="Operación falló")
    error: ErrorDetail = Field(..., description="Detalles del error")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Esquemas de respuesta para endpoints específicos
class ImageDetectionResponse(SuccessResponse):
    """Respuesta específica para detección en imágenes"""
    data: ImageProcessingResult = Field(..., description="Resultado de procesamiento")


class HealthCheckResponse(BaseModel):
    """Respuesta de health check"""
    status: str = Field(..., description="Estado del servicio")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    service: str = Field(..., description="Nombre del servicio")
    version: str = Field(..., description="Versión del servicio")