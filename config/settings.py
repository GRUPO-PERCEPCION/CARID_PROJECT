from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import List
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
    model_confidence_threshold: float = Field(default=0.5, env="MODEL_CONFIDENCE_THRESHOLD")
    model_iou_threshold: float = Field(default=0.4, env="MODEL_IOU_THRESHOLD")

    # Configuración CUDA/GPU
    use_gpu: bool = Field(default=True, env="USE_GPU")
    gpu_device: int = Field(default=0, env="GPU_DEVICE")
    model_device: str = Field(default="cuda:0", env="MODEL_DEVICE")

    # 🚀 CONFIGURACIÓN DE ARCHIVOS ACTUALIZADA - 150MB
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    static_dir: str = Field(default="./static", env="STATIC_DIR")
    max_file_size: int = Field(default=150, env="MAX_FILE_SIZE")  # ✅ AUMENTADO A 150MB
    allowed_extensions: str = Field(default="jpg,jpeg,png,mp4,avi,mov,mkv,webm", env="ALLOWED_EXTENSIONS")

    # Configuración de procesamiento de imágenes
    image_max_size: int = Field(default=1920, env="IMAGE_MAX_SIZE")

    # 🎬 CONFIGURACIÓN DE VIDEOS OPTIMIZADA
    max_video_duration: int = Field(default=600, env="MAX_VIDEO_DURATION")  # ✅ 10 minutos para videos grandes
    video_frame_skip: int = Field(default=3, env="VIDEO_FRAME_SKIP")
    video_min_detection_frames: int = Field(default=3, env="VIDEO_MIN_DETECTION_FRAMES")  # ✅ Más restrictivo
    video_similarity_threshold: float = Field(default=0.8, env="VIDEO_SIMILARITY_THRESHOLD")  # ✅ Más estricto
    video_max_tracking_distance: int = Field(default=8, env="VIDEO_MAX_TRACKING_DISTANCE")  # ✅ Mayor distancia
    video_processing_timeout: int = Field(default=1200, env="VIDEO_PROCESSING_TIMEOUT")  # ✅ 20 min timeout

    # 🔧 NUEVAS CONFIGURACIONES PARA TRACKING AVANZADO
    plate_confidence_weight: float = Field(default=0.4, env="PLATE_CONFIDENCE_WEIGHT")  # Peso detector placas
    char_confidence_weight: float = Field(default=0.6, env="CHAR_CONFIDENCE_WEIGHT")  # Peso reconocedor caracteres
    min_combined_confidence: float = Field(default=0.3, env="MIN_COMBINED_CONFIDENCE")  # Confianza mínima combinada
    tracking_iou_threshold: float = Field(default=0.2, env="TRACKING_IOU_THRESHOLD")  # IoU para tracking
    stability_frames_required: int = Field(default=5, env="STABILITY_FRAMES_REQUIRED")  # Frames para estabilidad

    # 🌐 CONFIGURACIÓN DE STREAMING EN TIEMPO REAL (NUEVO)
    streaming_enabled: bool = Field(default=True, env="STREAMING_ENABLED")
    max_websocket_connections: int = Field(default=20, env="MAX_WEBSOCKET_CONNECTIONS")
    websocket_ping_interval: int = Field(default=30, env="WEBSOCKET_PING_INTERVAL")
    websocket_ping_timeout: int = Field(default=10, env="WEBSOCKET_PING_TIMEOUT")

    # Frame processing para streaming
    streaming_frame_quality: int = Field(default=75, env="STREAMING_FRAME_QUALITY")  # Calidad JPEG 1-100
    streaming_frame_max_size: int = Field(default=800, env="STREAMING_FRAME_MAX_SIZE")  # Ancho máximo en pixels
    streaming_send_interval: float = Field(default=0.5, env="STREAMING_SEND_INTERVAL")  # Segundos entre envíos
    streaming_buffer_size: int = Field(default=10, env="STREAMING_BUFFER_SIZE")  # Frames en buffer

    # Optimización de streaming
    streaming_compression_enabled: bool = Field(default=True, env="STREAMING_COMPRESSION_ENABLED")
    streaming_adaptive_quality: bool = Field(default=True, env="STREAMING_ADAPTIVE_QUALITY")
    streaming_throttle_enabled: bool = Field(default=True, env="STREAMING_THROTTLE_ENABLED")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/api.log", env="LOG_FILE")

    # ✅ MÉTODOS ACTUALIZADOS PARA STREAMING EN TIEMPO REAL

    def get_streaming_config(self) -> dict:
        """Configuración específica para streaming en tiempo real"""
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
                "frame_skip": max(1, self.video_frame_skip - 1),  # Más frames para streaming
                "min_detection_frames": max(1, self.video_min_detection_frames - 1)
            }
        }

    def get_video_processing_config(self) -> dict:
        """Retorna configuración optimizada para procesamiento de video"""
        return {
            "max_duration": self.max_video_duration,
            "frame_skip": self.video_frame_skip,
            "min_detection_frames": self.video_min_detection_frames,
            "similarity_threshold": self.video_similarity_threshold,
            "max_tracking_distance": self.video_max_tracking_distance,
            "processing_timeout": self.video_processing_timeout,
            "confidence_threshold": max(0.25, self.model_confidence_threshold - 0.15),  # Más permisivo
            "iou_threshold": self.model_iou_threshold,
            "supported_formats": self.video_extensions_list,
            # 🆕 Nuevas configuraciones de tracking
            "plate_confidence_weight": self.plate_confidence_weight,
            "char_confidence_weight": self.char_confidence_weight,
            "min_combined_confidence": self.min_combined_confidence,
            "tracking_iou_threshold": self.tracking_iou_threshold,
            "stability_frames_required": self.stability_frames_required
        }

    def get_tracking_config(self) -> dict:
        """Configuración específica para tracking de placas"""
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
            f"{self.static_dir}/videos",  # NUEVO: para videos procesados
            f"{self.static_dir}/frames",  # NUEVO: para frames extraídos
            f"{self.static_dir}/streaming",  # NUEVO: para streaming frames
            os.path.dirname(self.log_file) if self.log_file else "./logs"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def validate_model_files(self) -> dict:
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

    def validate_video_settings(self) -> dict:
        """Valida configuraciones específicas de video"""
        validation = {
            "max_duration_valid": 10 <= self.max_video_duration <= 600,
            "frame_skip_valid": 1 <= self.video_frame_skip <= 10,
            "min_frames_valid": 1 <= self.video_min_detection_frames <= 10,
            "similarity_threshold_valid": 0.1 <= self.video_similarity_threshold <= 1.0,
            "tracking_distance_valid": 1 <= self.video_max_tracking_distance <= 20,
            "timeout_valid": 60 <= self.video_processing_timeout <= 1800
        }

        validation["all_valid"] = all(validation.values())
        return validation

    def validate_streaming_settings(self) -> dict:
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