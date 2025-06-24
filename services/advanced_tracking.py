"""
Sistema de Tracking Avanzado para Placas Vehiculares
Maneja las confianzas independientes de detección y reconocimiento
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from loguru import logger
from config.settings import settings


@dataclass
class PlateDetection:
    """Representa una detección individual de placa en un frame"""
    frame_num: int
    plate_bbox: List[float]
    plate_confidence: float  # Confianza del detector de placas
    plate_text: str
    char_confidence: float  # Confianza del reconocedor de caracteres
    char_count: int
    is_valid_format: bool
    combined_confidence: float = 0.0
    bbox_area: float = 0.0
    aspect_ratio: float = 0.0

    def __post_init__(self):
        """Calcula métricas derivadas"""
        if not self.combined_confidence:
            self.combined_confidence = self._calculate_combined_confidence()

        if not self.bbox_area:
            self.bbox_area = self._calculate_bbox_area()

        if not self.aspect_ratio:
            self.aspect_ratio = self._calculate_aspect_ratio()

    def _calculate_combined_confidence(self) -> float:
        """Calcula confianza combinada usando pesos configurables"""
        config = settings.get_tracking_config()
        plate_weight = config["confidence_weights"]["plate_detection"]
        char_weight = config["confidence_weights"]["character_recognition"]

        return (self.plate_confidence * plate_weight) + (self.char_confidence * char_weight)

    def _calculate_bbox_area(self) -> float:
        """Calcula área del bounding box"""
        x1, y1, x2, y2 = self.plate_bbox
        return abs(x2 - x1) * abs(y2 - y1)

    def _calculate_aspect_ratio(self) -> float:
        """Calcula aspect ratio del bounding box"""
        x1, y1, x2, y2 = self.plate_bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return width / height if height > 0 else 0.0


@dataclass
class AdvancedPlateTracker:
    """Tracker avanzado que maneja doble confianza y estabilidad temporal"""
    plate_text: str
    first_detection: PlateDetection

    # Métricas de tracking
    detection_count: int = 1
    first_seen: int = field(init=False)
    last_seen: int = field(init=False)

    # Mejor detección
    best_detection: PlateDetection = field(init=False)
    best_combined_confidence: float = field(init=False)
    best_plate_confidence: float = field(init=False)
    best_char_confidence: float = field(init=False)

    # Historial
    detections: List[PlateDetection] = field(default_factory=list)
    plate_confidences: List[float] = field(default_factory=list)
    char_confidences: List[float] = field(default_factory=list)
    combined_confidences: List[float] = field(default_factory=list)
    bbox_history: List[List[float]] = field(default_factory=list)

    # Métricas calculadas
    avg_plate_confidence: float = 0.0
    avg_char_confidence: float = 0.0
    avg_combined_confidence: float = 0.0
    stability_score: float = 0.0
    tracking_quality: str = "unknown"

    def __post_init__(self):
        """Inicializa el tracker con la primera detección"""
        self.first_seen = self.first_detection.frame_num
        self.last_seen = self.first_detection.frame_num
        self.best_detection = self.first_detection
        self.best_combined_confidence = self.first_detection.combined_confidence
        self.best_plate_confidence = self.first_detection.plate_confidence
        self.best_char_confidence = self.first_detection.char_confidence

        # Agregar primera detección al historial
        self.add_detection(self.first_detection)

    def add_detection(self, detection: PlateDetection) -> bool:
        """
        Agrega nueva detección al tracker

        Returns:
            True si la detección fue agregada, False si fue rechazada
        """
        config = settings.get_tracking_config()

        # Verificar que la detección cumple criterios mínimos
        if detection.combined_confidence < config["min_combined_confidence"]:
            logger.debug(f"🔍 Detección rechazada por baja confianza: {detection.combined_confidence:.3f}")
            return False

        # Verificar distancia temporal
        frame_distance = detection.frame_num - self.last_seen
        if frame_distance > config["max_tracking_distance"]:
            logger.debug(f"🔍 Detección rechazada por distancia temporal: {frame_distance} frames")
            return False

        # Verificar similitud espacial si no es la primera detección
        if self.bbox_history and not self._is_spatially_similar(detection.plate_bbox):
            logger.debug(f"🔍 Detección rechazada por posición muy diferente")
            return False

        # Agregar detección
        self.detections.append(detection)
        self.detection_count += 1
        self.last_seen = detection.frame_num

        # Actualizar historiales
        self.plate_confidences.append(detection.plate_confidence)
        self.char_confidences.append(detection.char_confidence)
        self.combined_confidences.append(detection.combined_confidence)
        self.bbox_history.append(detection.plate_bbox)

        # Actualizar mejores detecciones
        self._update_best_detections(detection)

        # Recalcular métricas
        self._recalculate_metrics()

        logger.debug(f"✅ Detección agregada: {detection.plate_text} "
                     f"(Frame: {detection.frame_num}, Conf: {detection.combined_confidence:.3f})")

        return True

    def _is_spatially_similar(self, new_bbox: List[float]) -> bool:
        """Verifica si la nueva detección está espacialmente cerca"""
        config = settings.get_tracking_config()

        if not self.bbox_history:
            return True

        # Calcular IoU con las últimas 3 detecciones
        recent_bboxes = self.bbox_history[-3:]
        max_iou = 0.0

        for bbox in recent_bboxes:
            iou = self._calculate_iou(bbox, new_bbox)
            max_iou = max(max_iou, iou)

        return max_iou > config["iou_threshold"]

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calcula Intersection over Union entre dos bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersección
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

    def _update_best_detections(self, detection: PlateDetection):
        """Actualiza las mejores detecciones por diferentes criterios"""

        # Mejor detección combinada
        if detection.combined_confidence > self.best_combined_confidence:
            self.best_detection = detection
            self.best_combined_confidence = detection.combined_confidence

        # Mejor confianza de placa
        if detection.plate_confidence > self.best_plate_confidence:
            self.best_plate_confidence = detection.plate_confidence

        # Mejor confianza de caracteres
        if detection.char_confidence > self.best_char_confidence:
            self.best_char_confidence = detection.char_confidence

    def _recalculate_metrics(self):
        """Recalcula todas las métricas del tracker"""
        if not self.detections:
            return

        # Promedios
        self.avg_plate_confidence = np.mean(self.plate_confidences)
        self.avg_char_confidence = np.mean(self.char_confidences)
        self.avg_combined_confidence = np.mean(self.combined_confidences)

        # Score de estabilidad
        self.stability_score = self._calculate_stability_score()

        # Calidad de tracking
        self.tracking_quality = self._determine_tracking_quality()

    def _calculate_stability_score(self) -> float:
        """Calcula score de estabilidad basado en consistencia"""
        if len(self.combined_confidences) < 2:
            return 0.5

        # Consistencia de confianza
        conf_std = np.std(self.combined_confidences)
        conf_mean = np.mean(self.combined_confidences)
        conf_consistency = 1.0 - min(conf_std / conf_mean, 1.0) if conf_mean > 0 else 0.0

        # Frecuencia de detección
        duration = self.last_seen - self.first_seen + 1
        detection_frequency = min(self.detection_count / max(duration, 1), 1.0)

        # Consistencia espacial (variación de área de bbox)
        if len(self.detections) > 1:
            areas = [det.bbox_area for det in self.detections]
            area_std = np.std(areas)
            area_mean = np.mean(areas)
            spatial_consistency = 1.0 - min(area_std / area_mean, 1.0) if area_mean > 0 else 0.0
        else:
            spatial_consistency = 1.0

        # Score final ponderado
        stability = (
                conf_consistency * 0.4 +
                detection_frequency * 0.3 +
                spatial_consistency * 0.3
        )

        return max(0.0, min(1.0, stability))

    def _determine_tracking_quality(self) -> str:
        """Determina la calidad del tracking"""
        config = settings.get_tracking_config()

        if self.detection_count < config["min_detection_frames"]:
            return "insufficient"
        elif self.stability_score > 0.8 and self.avg_combined_confidence > 0.6:
            return "excellent"
        elif self.stability_score > 0.6 and self.avg_combined_confidence > 0.4:
            return "good"
        elif self.stability_score > 0.4 and self.avg_combined_confidence > 0.3:
            return "fair"
        else:
            return "poor"

    def is_stable(self) -> bool:
        """Verifica si el tracker es estable y confiable"""
        config = settings.get_tracking_config()

        return (
                self.detection_count >= config["stability_frames_required"] and
                self.stability_score > 0.5 and
                self.avg_combined_confidence > config["min_combined_confidence"] and
                self.tracking_quality in ["excellent", "good", "fair"]
        )

    def get_best_frame_info(self) -> Dict[str, Any]:
        """Retorna información del mejor frame detectado"""
        return {
            "frame_num": self.best_detection.frame_num,
            "plate_text": self.best_detection.plate_text,
            "plate_confidence": self.best_detection.plate_confidence,
            "char_confidence": self.best_detection.char_confidence,
            "combined_confidence": self.best_detection.combined_confidence,
            "bbox": self.best_detection.plate_bbox,
            "is_valid_format": self.best_detection.is_valid_format,
            "char_count": self.best_detection.char_count
        }

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen completo del tracker"""
        return {
            "plate_text": self.plate_text,
            "detection_count": self.detection_count,
            "duration_frames": self.last_seen - self.first_seen + 1,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,

            # Mejores detecciones
            "best_frame": self.get_best_frame_info(),
            "best_combined_confidence": self.best_combined_confidence,
            "best_plate_confidence": self.best_plate_confidence,
            "best_char_confidence": self.best_char_confidence,

            # Promedios
            "avg_plate_confidence": round(self.avg_plate_confidence, 3),
            "avg_char_confidence": round(self.avg_char_confidence, 3),
            "avg_combined_confidence": round(self.avg_combined_confidence, 3),

            # Calidad
            "stability_score": round(self.stability_score, 3),
            "tracking_quality": self.tracking_quality,
            "is_stable": self.is_stable(),
            "is_valid_format": self.best_detection.is_valid_format,

            # Estadísticas adicionales
            "confidence_range": {
                "plate_min": min(self.plate_confidences),
                "plate_max": max(self.plate_confidences),
                "char_min": min(self.char_confidences),
                "char_max": max(self.char_confidences)
            }
        }

    def _are_texts_similar(self, text1: str, text2: str) -> bool:
        """Verifica similitud entre textos de placas"""
        if not text1 or not text2:
            return False

        # Normalizar
        text1 = text1.upper().replace('-', '').replace(' ', '')
        text2 = text2.upper().replace('-', '').replace(' ', '')

        if text1 == text2:
            return True

        # Similitud por caracteres
        if len(text1) == len(text2):
            differences = sum(c1 != c2 for c1, c2 in zip(text1, text2))
            similarity = 1 - (differences / len(text1))
            return similarity >= settings.get_tracking_config()["similarity_threshold"]

        return False


class AdvancedTrackingManager:
    """Gestor avanzado de múltiples trackers"""

    def __init__(self):
        self.active_trackers: Dict[str, AdvancedPlateTracker] = {}
        self.completed_trackers: List[AdvancedPlateTracker] = []
        self.config = settings.get_tracking_config()

    def process_frame_detections(self, frame_detections: List[Dict[str, Any]], frame_num: int):
        """Procesa las detecciones de un frame y actualiza trackers"""

        # Convertir detecciones a objetos PlateDetection
        plate_detections = []
        for det in frame_detections:
            plate_det = PlateDetection(
                frame_num=frame_num,
                plate_bbox=det["plate_bbox"],
                plate_confidence=det["plate_confidence"],
                plate_text=det["plate_text"],
                char_confidence=det.get("char_confidence", det.get("confidence", 0.0)),
                char_count=det.get("char_count", len(det["plate_text"])),
                is_valid_format=det.get("is_valid_plate", False)
            )
            plate_detections.append(plate_det)

        # Intentar asignar cada detección a un tracker existente
        unassigned_detections = []

        for detection in plate_detections:
            assigned = False

            # Buscar tracker compatible
            for tracker_id, tracker in self.active_trackers.items():
                if self._can_assign_to_tracker(detection, tracker):
                    if tracker.add_detection(detection):
                        assigned = True
                        logger.debug(f"✅ Detección asignada a tracker existente: {tracker_id}")
                        break

            if not assigned:
                unassigned_detections.append(detection)

        # Crear nuevos trackers para detecciones no asignadas
        for detection in unassigned_detections:
            tracker_id = f"{detection.plate_text}_{frame_num}"
            new_tracker = AdvancedPlateTracker(
                plate_text=detection.plate_text,
                first_detection=detection
            )
            self.active_trackers[tracker_id] = new_tracker
            logger.debug(f"🆕 Nuevo tracker creado: {tracker_id}")

        # Limpiar trackers inactivos
        self._cleanup_inactive_trackers(frame_num)

    def _can_assign_to_tracker(self, detection: PlateDetection, tracker: AdvancedPlateTracker) -> bool:
        """Determina si una detección puede asignarse a un tracker"""

        # Verificar similitud de texto
        if not tracker._are_texts_similar(detection.plate_text, tracker.plate_text):
            return False

        # Verificar distancia temporal
        frame_distance = detection.frame_num - tracker.last_seen
        if frame_distance > self.config["max_tracking_distance"]:
            return False

        # Verificar similitud espacial
        if tracker.bbox_history and not tracker._is_spatially_similar(detection.plate_bbox):
            return False

        return True

    def _cleanup_inactive_trackers(self, current_frame: int):
        """Mueve trackers inactivos a completados"""
        inactive_trackers = []

        for tracker_id, tracker in self.active_trackers.items():
            frame_distance = current_frame - tracker.last_seen

            if frame_distance > self.config["max_tracking_distance"]:
                inactive_trackers.append(tracker_id)
                self.completed_trackers.append(tracker)
                logger.debug(f"📝 Tracker movido a completados: {tracker_id}")

        # Eliminar trackers inactivos
        for tracker_id in inactive_trackers:
            del self.active_trackers[tracker_id]

    def get_final_results(self) -> List[Dict[str, Any]]:
        """Obtiene resultados finales de todos los trackers"""

        # Mover trackers activos restantes a completados
        for tracker in self.active_trackers.values():
            self.completed_trackers.append(tracker)

        # Filtrar solo trackers estables
        stable_trackers = [t for t in self.completed_trackers if t.is_stable()]

        # Ordenar por calidad combinada
        stable_trackers.sort(
            key=lambda t: (t.best_combined_confidence * 0.6 + t.stability_score * 0.4),
            reverse=True
        )

        # Generar resúmenes
        results = [tracker.get_summary() for tracker in stable_trackers]

        logger.info(f"📊 Tracking finalizado: {len(results)} placas estables de "
                    f"{len(self.completed_trackers)} trackers totales")

        return results