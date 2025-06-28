from loguru import logger
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import List, Dict, Any
import torch
import os


class Settings(BaseSettings):
    """Configuración de la aplicación desde variables de entorno"""

    # Configurar Pydantic para evitar warnings
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="allow",
        protected_namespaces=('settings_',)
    )

    # Configuración de la aplicación
    app_name: str = Field(default="CARID-ALPR-API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    # Configuración de modelos
    plate_model_path: str = Field(default="./models_trained/plate_detection.pt", env="PLATE_MODEL_PATH")
    char_model_path: str = Field(default="./models_trained/char_recognition.pt", env="CHAR_MODEL_PATH")
    model_confidence_threshold: float = Field(default=0.7, env="MODEL_CONFIDENCE_THRESHOLD")
    model_iou_threshold: float = Field(default=0.5, env="MODEL_IOU_THRESHOLD")

    # Configuración CUDA/GPU
    use_gpu: bool = Field(default=True, env="USE_GPU")
    gpu_device: int = Field(default=0, env="GPU_DEVICE")
    model_device: str = Field(default="cuda:0", env="MODEL_DEVICE")

    # Configuración de archivos
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    static_dir: str = Field(default="./static", env="STATIC_DIR")
    max_file_size: int = Field(default=150, env="MAX_FILE_SIZE")  # MB
    allowed_extensions: str = Field(default="jpg,jpeg,png,mp4,avi,mov,mkv,webm", env="ALLOWED_EXTENSIONS")

    # Configuración de procesamiento de imágenes
    image_max_size: int = Field(default=1920, env="IMAGE_MAX_SIZE")

    # ✅ CONFIGURACIÓN DE DETECCIÓN DE IMÁGENES
    image_confidence_threshold: float = Field(default=0.6, env="IMAGE_CONFIDENCE_THRESHOLD")
    image_iou_threshold: float = Field(default=0.5, env="IMAGE_IOU_THRESHOLD")
    image_max_detections: int = Field(default=10, env="IMAGE_MAX_DETECTIONS")
    image_enhance_image: bool = Field(default=True, env="IMAGE_ENHANCE_IMAGE")
    image_return_visualization: bool = Field(default=True, env="IMAGE_RETURN_VISUALIZATION")
    image_save_results: bool = Field(default=True, env="IMAGE_SAVE_RESULTS")

    # ✅ CONFIGURACIÓN DE DETECCIÓN RÁPIDA
    quick_confidence_threshold: float = Field(default=0.5, env="QUICK_CONFIDENCE_THRESHOLD")
    quick_iou_threshold: float = Field(default=0.5, env="QUICK_IOU_THRESHOLD")
    quick_max_detections: int = Field(default=5, env="QUICK_MAX_DETECTIONS")
    quick_enhance_image: bool = Field(default=False, env="QUICK_ENHANCE_IMAGE")
    quick_return_visualization: bool = Field(default=False, env="QUICK_RETURN_VISUALIZATION")
    quick_save_results: bool = Field(default=False, env="QUICK_SAVE_RESULTS")
    quick_frame_skip: int = Field(default=2, env="QUICK_FRAME_SKIP")
    quick_max_duration: int = Field(default=60, env="QUICK_MAX_DURATION")

    # ✅ CONFIGURACIÓN DE VIDEOS
    video_confidence_threshold: float = Field(default=0.5, env="VIDEO_CONFIDENCE_THRESHOLD")
    video_iou_threshold: float = Field(default=0.5, env="VIDEO_IOU_THRESHOLD")
    video_frame_skip: int = Field(default=3, env="VIDEO_FRAME_SKIP")
    video_max_duration: int = Field(default=600, env="VIDEO_MAX_DURATION")  # 10 minutos
    video_min_detection_frames: int = Field(default=3, env="VIDEO_MIN_DETECTION_FRAMES")
    video_save_results: bool = Field(default=True, env="VIDEO_SAVE_RESULTS")
    video_save_best_frames: bool = Field(default=True, env="VIDEO_SAVE_BEST_FRAMES")
    video_create_annotated_video: bool = Field(default=False, env="VIDEO_CREATE_ANNOTATED_VIDEO")
    video_processing_timeout: int = Field(default=1200, env="VIDEO_PROCESSING_TIMEOUT")  # 20 minutos

    # ✅ CONFIGURACIÓN DE TRACKING
    video_similarity_threshold: float = Field(default=0.8, env="VIDEO_SIMILARITY_THRESHOLD")
    video_max_tracking_distance: int = Field(default=8, env="VIDEO_MAX_TRACKING_DISTANCE")
    tracking_iou_threshold: float = Field(default=0.2, env="TRACKING_IOU_THRESHOLD")
    stability_frames_required: int = Field(default=5, env="STABILITY_FRAMES_REQUIRED")

    # ✅ CONFIGURACIÓN DE DETECTOR DE PLACAS
    plate_min_area: int = Field(default=500, env="PLATE_MIN_AREA")
    plate_max_detections: int = Field(default=10, env="PLATE_MAX_DETECTIONS")
    plate_min_aspect_ratio: float = Field(default=1.5, env="PLATE_MIN_ASPECT_RATIO")
    plate_max_aspect_ratio: float = Field(default=6.0, env="PLATE_MAX_ASPECT_RATIO")

    # ✅ CONFIGURACIÓN DE RECONOCEDOR DE CARACTERES
    char_min_confidence: float = Field(default=0.3, env="CHAR_MIN_CONFIDENCE")
    char_expected_count: int = Field(default=6, env="CHAR_EXPECTED_COUNT")
    char_max_characters: int = Field(default=10, env="CHAR_MAX_CHARACTERS")
    char_force_six_characters: bool = Field(default=True, env="CHAR_FORCE_SIX_CHARACTERS")
    char_strict_validation: bool = Field(default=False, env="CHAR_STRICT_VALIDATION")

    # ✅ CONFIGURACIÓN DE ROI
    roi_enabled: bool = Field(default=True, env="ROI_ENABLED")
    roi_percentage: float = Field(default=60.0, env="ROI_PERCENTAGE")  # 60% del centro de la imagen

    # ✅ CONFIGURACIÓN DE VALIDACIÓN (RANGOS)
    min_confidence_range: float = Field(default=0.1, env="MIN_CONFIDENCE_RANGE")
    max_confidence_range: float = Field(default=1.0, env="MAX_CONFIDENCE_RANGE")
    min_iou_range: float = Field(default=0.1, env="MIN_IOU_RANGE")
    max_iou_range: float = Field(default=1.0, env="MAX_IOU_RANGE")
    min_max_detections: int = Field(default=1, env="MIN_MAX_DETECTIONS")
    max_max_detections: int = Field(default=20, env="MAX_MAX_DETECTIONS")
    min_frame_skip: int = Field(default=1, env="MIN_FRAME_SKIP")
    max_frame_skip: int = Field(default=10, env="MAX_FRAME_SKIP")
    min_video_duration: int = Field(default=1, env="MIN_VIDEO_DURATION")
    max_video_duration_range: int = Field(default=1800, env="MAX_VIDEO_DURATION_RANGE")

    # ✅ CONFIGURACIÓN DE ADVERTENCIAS
    low_confidence_warning: float = Field(default=0.3, env="LOW_CONFIDENCE_WARNING")
    high_confidence_warning: float = Field(default=0.9, env="HIGH_CONFIDENCE_WARNING")
    high_frame_skip_warning: int = Field(default=5, env="HIGH_FRAME_SKIP_WARNING")
    long_video_warning: int = Field(default=300, env="LONG_VIDEO_WARNING")

    # ✅ CONFIGURACIÓN DE RECOMENDACIONES
    recommended_confidence_min: float = Field(default=0.4, env="RECOMMENDED_CONFIDENCE_MIN")
    recommended_confidence_max: float = Field(default=0.8, env="RECOMMENDED_CONFIDENCE_MAX")
    recommended_frame_skip_min: int = Field(default=2, env="RECOMMENDED_FRAME_SKIP_MIN")
    recommended_frame_skip_max: int = Field(default=4, env="RECOMMENDED_FRAME_SKIP_MAX")

    # ✅ CONFIGURACIÓN DE STREAMING
    streaming_enabled: bool = Field(default=True, env="STREAMING_ENABLED")
    max_websocket_connections: int = Field(default=20, env="MAX_WEBSOCKET_CONNECTIONS")
    websocket_ping_interval: int = Field(default=30, env="WEBSOCKET_PING_INTERVAL")
    websocket_ping_timeout: int = Field(default=10, env="WEBSOCKET_PING_TIMEOUT")
    streaming_frame_quality: int = Field(default=75, env="STREAMING_FRAME_QUALITY")
    streaming_frame_max_size: int = Field(default=800, env="STREAMING_FRAME_MAX_SIZE")
    streaming_send_interval: float = Field(default=0.5, env="STREAMING_SEND_INTERVAL")
    streaming_buffer_size: int = Field(default=10, env="STREAMING_BUFFER_SIZE")
    streaming_compression_enabled: bool = Field(default=True, env="STREAMING_COMPRESSION_ENABLED")
    streaming_adaptive_quality: bool = Field(default=True, env="STREAMING_ADAPTIVE_QUALITY")
    streaming_throttle_enabled: bool = Field(default=True, env="STREAMING_THROTTLE_ENABLED")

    # ✅ CONFIGURACIÓN DE CONFIANZA COMBINADA
    plate_confidence_weight: float = Field(default=0.4, env="PLATE_CONFIDENCE_WEIGHT")
    char_confidence_weight: float = Field(default=0.6, env="CHAR_CONFIDENCE_WEIGHT")
    min_combined_confidence: float = Field(default=0.3, env="MIN_COMBINED_CONFIDENCE")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/api.log", env="LOG_FILE")

    # ✅ MÉTODOS DE CONFIGURACIÓN CENTRALIZADOS

    def get_image_detection_config(self) -> Dict[str, Any]:
        """Configuración para detección en imágenes"""
        return {
            "confidence_threshold": self.image_confidence_threshold,
            "iou_threshold": self.image_iou_threshold,
            "max_detections": self.image_max_detections,
            "enhance_image": self.image_enhance_image,
            "return_visualization": self.image_return_visualization,
            "save_results": self.image_save_results
        }

    def get_quick_detection_config(self) -> Dict[str, Any]:
        """Configuración para detección rápida"""
        return {
            "confidence_threshold": self.quick_confidence_threshold,
            "iou_threshold": self.quick_iou_threshold,
            "max_detections": self.quick_max_detections,
            "enhance_image": self.quick_enhance_image,
            "return_visualization": self.quick_return_visualization,
            "save_results": self.quick_save_results,
            "frame_skip": self.quick_frame_skip,
            "max_duration": self.quick_max_duration
        }

    def get_video_detection_config(self) -> Dict[str, Any]:
        """Configuración para detección en videos"""
        return {
            "confidence_threshold": self.video_confidence_threshold,
            "iou_threshold": self.video_iou_threshold,
            "frame_skip": self.video_frame_skip,
            "max_duration": self.video_max_duration,
            "min_detection_frames": self.video_min_detection_frames,
            "save_results": self.video_save_results,
            "save_best_frames": self.video_save_best_frames,
            "create_annotated_video": self.video_create_annotated_video,
            "processing_timeout": self.video_processing_timeout
        }

    def get_tracking_config(self) -> Dict[str, Any]:
        """Configuración para tracking de placas"""
        return {
            "similarity_threshold": self.video_similarity_threshold,
            "min_detection_frames": self.video_min_detection_frames,
            "max_tracking_distance": self.video_max_tracking_distance,
            "iou_threshold": self.tracking_iou_threshold,
            "stability_frames_required": self.stability_frames_required,
            "confidence_weights": {
                "plate_detection": self.plate_confidence_weight,
                "character_recognition": self.char_confidence_weight
            },
            "min_combined_confidence": self.min_combined_confidence
        }

    def get_plate_detector_config(self) -> Dict[str, Any]:
        """Configuración para el detector de placas"""
        return {
            "min_plate_area": self.plate_min_area,
            "max_detections": self.plate_max_detections,
            "min_aspect_ratio": self.plate_min_aspect_ratio,
            "max_aspect_ratio": self.plate_max_aspect_ratio,
            "confidence_threshold": self.model_confidence_threshold,
            "iou_threshold": self.model_iou_threshold
        }

    def get_char_recognizer_config(self) -> Dict[str, Any]:
        """Configuración para el reconocedor de caracteres"""
        return {
            "min_char_confidence": self.char_min_confidence,
            "expected_char_count": self.char_expected_count,
            "max_characters": self.char_max_characters,
            "force_six_characters": self.char_force_six_characters,
            "strict_validation": self.char_strict_validation
        }

    def get_roi_config(self) -> Dict[str, Any]:
        """Configuración para ROI (Región de Interés)"""
        return {
            "enabled": self.roi_enabled,
            "percentage": self.roi_percentage
        }

    def get_validation_config(self) -> Dict[str, Any]:
        """Configuración para validación de parámetros"""
        return {
            "confidence_range": [self.min_confidence_range, self.max_confidence_range],
            "iou_range": [self.min_iou_range, self.max_iou_range],
            "max_detections_range": [self.min_max_detections, self.max_max_detections],
            "frame_skip_range": [self.min_frame_skip, self.max_frame_skip],
            "video_duration_range": [self.min_video_duration, self.max_video_duration_range],
            "warnings": {
                "low_confidence": self.low_confidence_warning,
                "high_confidence": self.high_confidence_warning,
                "high_frame_skip": self.high_frame_skip_warning,
                "long_video": self.long_video_warning
            },
            "recommendations": {
                "confidence_range": [self.recommended_confidence_min, self.recommended_confidence_max],
                "frame_skip_range": [self.recommended_frame_skip_min, self.recommended_frame_skip_max]
            }
        }

    def get_streaming_config(self) -> Dict[str, Any]:
        """Configuración para streaming en tiempo real"""
        return {
            "enabled": self.streaming_enabled,
            "websocket": {
                "max_connections": self.max_websocket_connections,
                "ping_interval": self.websocket_ping_interval,
                "ping_timeout": self.websocket_ping_timeout
            },
            "frame_processing": {
                "quality": self.streaming_frame_quality,
                "max_size": self.streaming_frame_max_size,
                "send_interval": self.streaming_send_interval,
                "buffer_size": self.streaming_buffer_size,
                "compression_enabled": self.streaming_compression_enabled,
                "adaptive_quality": self.streaming_adaptive_quality,
                "throttle_enabled": self.streaming_throttle_enabled
            },
            "detection": {
                "confidence_threshold": max(0.25, self.model_confidence_threshold - 0.15),
                "iou_threshold": self.model_iou_threshold,
                "frame_skip": max(1, self.video_frame_skip - 1),
                "min_detection_frames": max(1, self.video_min_detection_frames - 1)
            }
        }

    # ✅ PROPIEDADES ADICIONALES

    @property
    def force_six_characters(self) -> bool:
        """Alias para char_force_six_characters"""
        return self.char_force_six_characters

    @property
    def allowed_extensions_list(self) -> List[str]:
        """Retorna lista de extensiones permitidas"""
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def image_extensions_list(self) -> List[str]:
        """Retorna lista de extensiones de imágenes"""
        image_exts = ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]
        return [ext for ext in self.allowed_extensions_list if ext in image_exts]

    @property
    def video_extensions_list(self) -> List[str]:
        """Retorna lista de extensiones de videos"""
        video_exts = ["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"]
        return [ext for ext in self.allowed_extensions_list if ext in video_exts]

    @property
    def device(self) -> str:
        """Determina el dispositivo a usar (CUDA o CPU)"""
        if self.use_gpu and torch.cuda.is_available():
            return self.model_device
        return "cpu"

    @property
    def is_cuda_available(self) -> bool:
        """Verifica si CUDA está disponible"""
        return torch.cuda.is_available()

    def create_directories(self):
        """Crea los directorios necesarios si no existen"""
        directories = [
            self.upload_dir,
            self.static_dir,
            f"{self.upload_dir}/temp",
            f"{self.static_dir}/results",
            f"{self.static_dir}/videos",
            f"{self.static_dir}/frames",
            f"{self.static_dir}/streaming",
            os.path.dirname(self.log_file) if self.log_file else "./logs"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def validate_model_files(self) -> Dict[str, Any]:
        """Valida que los archivos de modelos existan"""
        validation = {
            "plate_model_exists": os.path.exists(self.plate_model_path),
            "char_model_exists": os.path.exists(self.char_model_path),
            "models_dir_exists": os.path.exists("./models_trained"),
            "plate_model_path": self.plate_model_path,
            "char_model_path": self.char_model_path,
            "plate_model_size": self._get_file_size(self.plate_model_path),
            "char_model_size": self._get_file_size(self.char_model_path)
        }
        return validation

    def validate_video_settings(self) -> Dict[str, Any]:
        """Valida configuraciones específicas de video"""
        validation = {
            "max_duration_valid": 10 <= self.video_max_duration <= 1800,
            "frame_skip_valid": 1 <= self.video_frame_skip <= 10,
            "min_frames_valid": 1 <= self.video_min_detection_frames <= 10,
            "similarity_threshold_valid": 0.1 <= self.video_similarity_threshold <= 1.0,
            "tracking_distance_valid": 1 <= self.video_max_tracking_distance <= 20,
            "timeout_valid": 60 <= self.video_processing_timeout <= 1800
        }

        validation["all_valid"] = all(validation.values())
        return validation

    def validate_streaming_settings(self) -> Dict[str, Any]:
        """Valida configuraciones específicas de streaming"""
        validation = {
            "streaming_enabled": self.streaming_enabled,
            "max_connections_valid": 1 <= self.max_websocket_connections <= 50,
            "frame_quality_valid": 10 <= self.streaming_frame_quality <= 100,
            "frame_size_valid": 320 <= self.streaming_frame_max_size <= 1920,
            "send_interval_valid": 0.1 <= self.streaming_send_interval <= 5.0,
            "buffer_size_valid": 1 <= self.streaming_buffer_size <= 50,
            "websockets_available": self._check_websockets_available()
        }

        validation["all_valid"] = all(validation.values())
        return validation

    def _check_websockets_available(self) -> bool:
        """Verifica que las dependencias de WebSocket estén disponibles"""
        try:
            import websockets
            return True
        except ImportError:
            logger.warning("⚠️ Librería 'websockets' no encontrada. Ejecuta: pip install websockets")
            return False

    def _get_file_size(self, file_path: str) -> float:
        """Obtiene el tamaño del archivo en MB"""
        try:
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                return round(size_bytes / (1024 * 1024), 2)
            return 0.0
        except Exception:
            return 0.0


# Instancia global de configuración
settings = Settings()