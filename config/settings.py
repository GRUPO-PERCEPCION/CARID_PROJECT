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
        protected_namespaces=('settings_',)  # Solución para warnings
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

    # Configuración de archivos
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    static_dir: str = Field(default="./static", env="STATIC_DIR")
    max_file_size: int = Field(default=50, env="MAX_FILE_SIZE")  # MB
    allowed_extensions: str = Field(default="jpg,jpeg,png,mp4,avi,mov", env="ALLOWED_EXTENSIONS")

    # Configuración de procesamiento
    max_video_duration: int = Field(default=300, env="MAX_VIDEO_DURATION")  # segundos
    frame_skip: int = Field(default=1, env="FRAME_SKIP")
    image_max_size: int = Field(default=1920, env="IMAGE_MAX_SIZE")  # píxeles

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/api.log", env="LOG_FILE")

    @property
    def allowed_extensions_list(self) -> List[str]:
        """Retorna lista de extensiones permitidas"""
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

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