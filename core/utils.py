import os
import uuid
from typing import List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from loguru import logger
from config.settings import settings


def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    """Genera un nombre de archivo único"""
    file_extension = Path(original_filename).suffix
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}{unique_id}{file_extension}"


def is_allowed_file(filename: str) -> bool:
    """Verifica si el archivo tiene una extensión permitida"""
    if not filename:
        return False

    file_extension = Path(filename).suffix[1:].lower()  # Remover el punto
    return file_extension in settings.allowed_extensions_list


def get_file_size_mb(file_path: str) -> float:
    """Obtiene el tamaño del archivo en MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def validate_file_size(file_path: str) -> bool:
    """Valida que el archivo no exceda el tamaño máximo"""
    file_size = get_file_size_mb(file_path)
    return file_size <= settings.max_file_size


def ensure_directory_exists(directory_path: str) -> bool:
    """Asegura que un directorio existe, lo crea si no existe"""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creando directorio {directory_path}: {str(e)}")
        return False


def clean_temp_files(directory: str, max_age_hours: int = 24):
    """Limpia archivos temporales antiguos"""
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    logger.info(f"Archivo temporal eliminado: {filename}")
    except Exception as e:
        logger.warning(f"Error limpiando archivos temporales: {str(e)}")


def resize_image_if_needed(image_path: str, max_size: int = None) -> str:
    """Redimensiona una imagen si excede el tamaño máximo"""
    if max_size is None:
        max_size = settings.image_max_size

    try:
        with Image.open(image_path) as img:
            original_size = img.size

            # Verificar si necesita redimensión
            if max(original_size) <= max_size:
                return image_path

            # Calcular nuevo tamaño manteniendo aspect ratio
            if original_size[0] > original_size[1]:
                new_width = max_size
                new_height = int((max_size * original_size[1]) / original_size[0])
            else:
                new_height = max_size
                new_width = int((max_size * original_size[0]) / original_size[1])

            # Redimensionar y guardar
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            resized_img.save(image_path, quality=95)

            logger.info(f"Imagen redimensionada de {original_size} a {new_width}x{new_height}")
            return image_path

    except Exception as e:
        logger.error(f"Error redimensionando imagen: {str(e)}")
        return image_path


def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """Obtiene las dimensiones de una imagen"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        logger.error(f"Error obteniendo dimensiones: {str(e)}")
        return None


def is_valid_image(image_path: str) -> bool:
    """Verifica si un archivo es una imagen válida"""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verifica que la imagen no esté corrupta
        return True
    except Exception:
        return False


def is_valid_video(video_path: str) -> bool:
    """Verifica si un archivo es un video válido"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        ret, frame = cap.read()
        cap.release()
        return ret
    except Exception:
        return False


def get_video_info(video_path: str) -> Optional[dict]:
    """Obtiene información básica de un video"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "size_mb": get_file_size_mb(video_path)
        }
    except Exception as e:
        logger.error(f"Error obteniendo info del video: {str(e)}")
        return None


def format_detection_confidence(confidence: float) -> str:
    """Formatea la confianza como porcentaje"""
    return f"{confidence * 100:.1f}%"


def calculate_bbox_area(bbox: List[float]) -> float:
    """Calcula el área de un bounding box [x1, y1, x2, y2]"""
    if len(bbox) != 4:
        return 0.0

    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width * height


def normalize_bbox_coordinates(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """Normaliza las coordenadas del bbox a valores entre 0 y 1"""
    if len(bbox) != 4:
        return bbox

    x1, y1, x2, y2 = bbox
    return [
        x1 / img_width,
        y1 / img_height,
        x2 / img_width,
        y2 / img_height
    ]


class PerformanceTimer:
    """Utilidad para medir tiempos de ejecución"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"⏱️ {self.name}: {duration:.3f}s")

    @property
    def duration(self) -> float:
        """Retorna la duración en segundos"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0