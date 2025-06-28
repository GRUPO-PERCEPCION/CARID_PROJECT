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


def get_video_frame_at_time(video_path: str, time_seconds: float) -> Optional[np.ndarray]:
    """
    Extrae un frame específico del video en un tiempo dado

    Args:
        video_path: Ruta del video
        time_seconds: Tiempo en segundos donde extraer el frame

    Returns:
        Frame como numpy array en formato RGB o None si hay error
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(time_seconds * fps)

        # Ir al frame específico
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        cap.release()

        if ret:
            # Convertir de BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb

        return None

    except Exception as e:
        logger.error(f"Error extrayendo frame en tiempo {time_seconds}s: {str(e)}")
        return None


def extract_video_frames(video_path: str, frame_skip: int = 1, max_frames: int = None) -> List[np.ndarray]:
    """
    Extrae frames de un video

    Args:
        video_path: Ruta del video
        frame_skip: Extraer cada N frames
        max_frames: Máximo número de frames a extraer

    Returns:
        Lista de frames como numpy arrays en formato RGB
    """
    frames = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames

        frame_num = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar solo cada N frames
            if frame_num % frame_skip == 0:
                # Convertir de BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1

                # Verificar límite máximo
                if max_frames and extracted_count >= max_frames:
                    break

            frame_num += 1

        cap.release()
        logger.info(f"Extraídos {len(frames)} frames del video")

    except Exception as e:
        logger.error(f"Error extrayendo frames: {str(e)}")

    return frames


def save_frame_as_image(frame: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """
    Guarda un frame como imagen

    Args:
        frame: Frame en formato RGB
        output_path: Ruta donde guardar la imagen
        quality: Calidad de compresión JPEG (1-100)

    Returns:
        True si se guardó exitosamente, False en caso contrario
    """
    try:
        # Convertir de RGB a BGR para OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Guardar imagen
        success = cv2.imwrite(output_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

        if success:
            logger.info(f"Frame guardado como imagen: {output_path}")

        return success

    except Exception as e:
        logger.error(f"Error guardando frame como imagen: {str(e)}")
        return False


def validate_video_duration(video_path: str, max_duration_seconds: int) -> bool:
    """
    Valida que la duración del video no exceda el límite

    Args:
        video_path: Ruta del video
        max_duration_seconds: Duración máxima permitida en segundos

    Returns:
        True si la duración es válida, False en caso contrario
    """
    try:
        video_info = get_video_info(video_path)
        if not video_info:
            return False

        return video_info["duration_seconds"] <= max_duration_seconds

    except Exception as e:
        logger.error(f"Error validando duración del video: {str(e)}")
        return False


def get_video_codec_info(video_path: str) -> Optional[dict]:
    """
    Obtiene información del codec del video

    Args:
        video_path: Ruta del video

    Returns:
        Diccionario con información del codec o None si hay error
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        # Obtener código del codec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + \
                chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)

        # Información adicional
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        return {
            "codec": codec,
            "fourcc": fourcc,
            "fps": fps,
            "frame_count": frame_count,
            "resolution": f"{width}x{height}",
            "aspect_ratio": round(width / height, 2) if height > 0 else 0
        }

    except Exception as e:
        logger.error(f"Error obteniendo información del codec: {str(e)}")
        return None


def estimate_processing_time(video_info: dict, frame_skip: int = 3) -> dict:
    """
    Estima el tiempo de procesamiento para un video

    Args:
        video_info: Información del video obtenida con get_video_info()
        frame_skip: Número de frames a saltar

    Returns:
        Diccionario con estimaciones de tiempo
    """
    try:
        if not video_info:
            return {"error": "Información de video inválida"}

        total_frames = video_info["frame_count"]
        frames_to_process = total_frames // frame_skip
        duration_seconds = video_info["duration_seconds"]

        # Estimaciones basadas en benchmarks (ajustar según hardware)
        # Tiempo base por frame procesado (en segundos)
        base_time_per_frame = 0.1  # 100ms por frame en promedio

        # Factores de ajuste
        resolution_factor = 1.0
        if video_info["width"] > 1920:  # 4K+
            resolution_factor = 2.0
        elif video_info["width"] > 1280:  # HD+
            resolution_factor = 1.5

        # Cálculo de tiempo estimado
        estimated_seconds = frames_to_process * base_time_per_frame * resolution_factor

        return {
            "total_frames": total_frames,
            "frames_to_process": frames_to_process,
            "estimated_processing_time_seconds": round(estimated_seconds, 1),
            "estimated_processing_time_minutes": round(estimated_seconds / 60, 1),
            "processing_speed_ratio": round(duration_seconds / estimated_seconds, 1),
            "resolution_factor": resolution_factor,
            "recommendation": _get_processing_recommendation(estimated_seconds, duration_seconds)
        }

    except Exception as e:
        logger.error(f"Error estimando tiempo de procesamiento: {str(e)}")
        return {"error": str(e)}


def _get_processing_recommendation(estimated_time: float, video_duration: float) -> str:
    """Genera recomendación basada en tiempo estimado"""
    ratio = estimated_time / video_duration

    if ratio < 0.5:
        return "Procesamiento muy rápido - óptimo"
    elif ratio < 1.0:
        return "Procesamiento rápido - bueno"
    elif ratio < 2.0:
        return "Procesamiento normal - aceptable"
    elif ratio < 5.0:
        return "Procesamiento lento - considerar aumentar frame_skip"
    else:
        return "Procesamiento muy lento - recomendado aumentar frame_skip o reducir resolución"


def create_video_thumbnail(video_path: str, output_path: str, time_seconds: float = None) -> bool:
    """
    Crea una miniatura del video

    Args:
        video_path: Ruta del video
        output_path: Ruta donde guardar la miniatura
        time_seconds: Tiempo específico para la miniatura (None = mitad del video)

    Returns:
        True si se creó exitosamente, False en caso contrario
    """
    try:
        video_info = get_video_info(video_path)
        if not video_info:
            return False

        # Si no se especifica tiempo, usar la mitad del video
        if time_seconds is None:
            time_seconds = video_info["duration_seconds"] / 2

        # Extraer frame
        frame = get_video_frame_at_time(video_path, time_seconds)
        if frame is None:
            return False

        # Redimensionar para thumbnail (máximo 320px de ancho)
        height, width = frame.shape[:2]
        if width > 320:
            new_width = 320
            new_height = int((320 * height) / width)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Guardar thumbnail
        return save_frame_as_image(frame, output_path, quality=85)

    except Exception as e:
        logger.error(f"Error creando thumbnail: {str(e)}")
        return False


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


def calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calcula Intersection over Union (IoU) entre dos bounding boxes

    Args:
        bbox1: Primera bbox [x1, y1, x2, y2]
        bbox2: Segunda bbox [x1, y1, x2, y2]

    Returns:
        Valor IoU entre 0 y 1
    """
    try:
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0

        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calcular intersección
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)

        if x_max <= x_min or y_max <= y_min:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    except Exception as e:
        logger.error(f"Error calculando IoU: {str(e)}")
        return 0.0


def filter_overlapping_detections(detections: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    """
    Filtra detecciones superpuestas usando Non-Maximum Suppression

    Args:
        detections: Lista de detecciones con 'bbox' y 'confidence'
        iou_threshold: Umbral IoU para considerar superposición

    Returns:
        Lista filtrada de detecciones
    """
    if not detections:
        return []

    try:
        # Ordenar por confianza (mayor a menor)
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)

        filtered = []

        for detection in sorted_detections:
            bbox = detection.get('bbox', [])
            if len(bbox) != 4:
                continue

            # Verificar si se superpone significativamente con alguna detección ya seleccionada
            overlaps = False
            for selected in filtered:
                selected_bbox = selected.get('bbox', [])
                if len(selected_bbox) == 4:
                    iou = calculate_bbox_iou(bbox, selected_bbox)
                    if iou > iou_threshold:
                        overlaps = True
                        break

            if not overlaps:
                filtered.append(detection)

        logger.info(f"Filtrado NMS: {len(detections)} -> {len(filtered)} detecciones")
        return filtered

    except Exception as e:
        logger.error(f"Error filtrando detecciones superpuestas: {str(e)}")
        return detections


def format_duration(seconds: float) -> str:
    """
    Formatea duración en segundos a formato legible

    Args:
        seconds: Duración en segundos

    Returns:
        String formateado (ej: "1m 30s", "45s", "2h 15m")
    """
    try:
        if seconds < 0:
            return "0s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    except Exception:
        return "N/A"


def get_memory_usage() -> dict:
    """
    Obtiene información de uso de memoria del sistema

    Returns:
        Diccionario con información de memoria
    """
    try:
        import psutil

        memory = psutil.virtual_memory()

        return {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "used_gb": round(memory.used / (1024 ** 3), 2),
            "percent_used": memory.percent,
            "free_gb": round(memory.free / (1024 ** 3), 2)
        }

    except ImportError:
        return {"error": "psutil no disponible"}
    except Exception as e:
        return {"error": str(e)}


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


class VideoProgressTracker:
    """Utilidad para trackear progreso de procesamiento de video"""

    def __init__(self, total_frames: int, name: str = "Video Processing"):
        self.name = name
        self.total_frames = total_frames
        self.processed_frames = 0
        self.start_time = None
        self.last_update = 0

    def start(self):
        """Inicia el tracking"""
        import time
        self.start_time = time.time()
        self.last_update = self.start_time
        logger.info(f"🎬 Iniciando {self.name}: {self.total_frames} frames totales")

    def update(self, frames_processed: int = 1):
        """Actualiza el progreso"""
        import time
        self.processed_frames += frames_processed
        current_time = time.time()

        # Log cada 30 segundos o al completar
        if current_time - self.last_update > 30 or self.processed_frames >= self.total_frames:
            progress = (self.processed_frames / self.total_frames) * 100
            elapsed = current_time - self.start_time

            if self.processed_frames > 0:
                estimated_total = elapsed * self.total_frames / self.processed_frames
                remaining = estimated_total - elapsed

                logger.info(f"📊 {self.name}: {progress:.1f}% "
                            f"({self.processed_frames}/{self.total_frames}) - "
                            f"Tiempo restante: {format_duration(remaining)}")

            self.last_update = current_time

    def finish(self):
        """Finaliza el tracking"""
        import time
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.success(f"✅ {self.name} completado en {format_duration(total_time)}")


# Funciones de utilidad específicas para el proyecto ALPR

def validate_peruvian_plate_format(plate_text: str) -> bool:
    """
    ✅ CORREGIDO: Valida formato de placa peruana
    Acepta tanto formatos con guión como sin guión (6 caracteres)
    """
    import re

    if not plate_text:
        return False

    # ✅ PATRONES ACTUALIZADOS: Con y sin guión
    patterns = [
        # Formatos CON guión (formateados)
        r'^[A-Z]{3}-\d{3}$',  # ABC-123
        r'^[A-Z]{2}-\d{4}$',  # AB-1234
        r'^[A-Z]\d{2}-\d{3}$',  # A12-345 (motos)

        # ✅ NUEVOS: Formatos SIN guión (como detecta el modelo)
        r'^[A-Z]{3}\d{3}$',  # ABC123
        r'^[A-Z]{2}\d{4}$',  # AB1234
        r'^[A-Z]\d{2}\d{3}$',  # A12345 (motos)
    ]

    for pattern in patterns:
        if re.match(pattern, plate_text.upper()):
            return True

    return False


def clean_plate_text(plate_text: str) -> str:
    """
    ✅ CORREGIDO: Limpia y normaliza texto de placa
    Mantiene solo caracteres alfanuméricos si no hay guión válido
    """
    if not plate_text:
        return ""

    # Convertir a mayúsculas
    cleaned = plate_text.upper()

    # ✅ NUEVA LÓGICA: Si ya tiene formato válido, mantenerlo
    if validate_peruvian_plate_format(cleaned):
        return cleaned

    # ✅ Si no es válido, limpiar solo alfanuméricos
    import re
    alphanumeric_only = re.sub(r'[^A-Z0-9]', '', cleaned)

    # ✅ Si son exactamente 6 caracteres, intentar formatear
    if len(alphanumeric_only) == 6:
        # ABC123 -> ABC-123
        if alphanumeric_only[:3].isalpha() and alphanumeric_only[3:].isdigit():
            return f"{alphanumeric_only[:3]}-{alphanumeric_only[3:]}"
        # AB1234 -> AB-1234
        elif alphanumeric_only[:2].isalpha() and alphanumeric_only[2:].isdigit():
            return f"{alphanumeric_only[:2]}-{alphanumeric_only[2:]}"

    # ✅ Si no se puede formatear, devolver sin guión
    return alphanumeric_only


def format_plate_text_for_display(raw_plate_text: str) -> str:
    """
    ✅ NUEVA FUNCIÓN: Formatea placa de 6 caracteres para mostrar con guión
    """
    if not raw_plate_text or len(raw_plate_text) != 6:
        return raw_plate_text

    # ABC123 -> ABC-123
    if raw_plate_text[:3].isalpha() and raw_plate_text[3:].isdigit():
        return f"{raw_plate_text[:3]}-{raw_plate_text[3:]}"

    # AB1234 -> AB-1234
    elif raw_plate_text[:2].isalpha() and raw_plate_text[2:].isdigit():
        return f"{raw_plate_text[:2]}-{raw_plate_text[2:]}"

    # Si no coincide con patrones, devolver sin cambios
    return raw_plate_text


def extract_raw_plate_text(formatted_plate_text: str) -> str:
    """
    ✅ NUEVA FUNCIÓN: Extrae texto crudo (sin guión) de texto formateado
    """
    if not formatted_plate_text:
        return ""

    # Remover guiones y espacios
    raw_text = ''.join(c for c in formatted_plate_text if c.isalnum()).upper()

    return raw_text


def is_six_char_plate_format(plate_text: str) -> bool:
    """
    ✅ NUEVA FUNCIÓN: Verifica si es formato válido de 6 caracteres
    """
    if not plate_text:
        return False

    # Remover guiones para verificar
    clean_text = extract_raw_plate_text(plate_text)

    if len(clean_text) != 6:
        return False

    # Verificar patrones válidos
    import re
    patterns = [
        r'^[A-Z]{3}\d{3}$',  # ABC123
        r'^[A-Z]{2}\d{4}$',  # AB1234
    ]

    return any(re.match(pattern, clean_text) for pattern in patterns)