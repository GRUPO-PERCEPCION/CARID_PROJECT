# services/real_time_video_service.py
"""
Servicio para procesamiento de video en tiempo real con WebSocket streaming
Integrado completamente con el sistema CARID existente
"""

import cv2
import numpy as np
import time
import asyncio
import base64
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from config.settings import settings
from models.model_manager import model_manager
from services.file_service import file_service
from api.websocket_manager import connection_manager, StreamingStatus
from core.utils import PerformanceTimer, get_video_info, format_duration


@dataclass
class FrameProcessingResult:
    """Resultado del procesamiento de un frame"""
    frame_num: int
    timestamp: float
    processing_time: float
    detections: List[Dict[str, Any]]
    frame_image_base64: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class RealTimeFrameProcessor:
    """Procesador de frames optimizado para tiempo real"""

    def __init__(self):
        self.model_manager = model_manager
        self.processing_config = settings.get_streaming_config()

    def process_frame_sync(
            self,
            frame: np.ndarray,
            frame_num: int,
            processing_params: Dict[str, Any]
    ) -> FrameProcessingResult:
        """
        Procesa un frame de forma síncrona optimizada para tiempo real

        Args:
            frame: Frame en formato RGB
            frame_num: Número del frame
            processing_params: Parámetros de procesamiento

        Returns:
            Resultado del procesamiento
        """
        start_time = time.time()

        try:
            # Extraer parámetros
            confidence_threshold = processing_params.get("confidence_threshold", 0.3)
            iou_threshold = processing_params.get("iou_threshold", 0.4)
            enhance_processing = processing_params.get("enhance_processing", False)

            # Procesar con el pipeline existente
            pipeline_result = self.model_manager.process_full_pipeline(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )

            # Extraer detecciones válidas
            detections = []
            if pipeline_result.get("success") and pipeline_result.get("final_results"):
                for i, plate_result in enumerate(pipeline_result["final_results"]):
                    if plate_result["plate_text"]:
                        detection = {
                            "detection_id": f"{frame_num}_{i}",
                            "frame_num": frame_num,
                            "timestamp": frame_num / 30.0,  # Estimado, se corregirá con FPS real
                            "plate_text": plate_result["plate_text"],
                            "plate_confidence": plate_result["plate_confidence"],
                            "char_confidence": plate_result.get("character_recognition", {}).get("confidence", 0.0),
                            "overall_confidence": plate_result["overall_confidence"],
                            "plate_bbox": plate_result["plate_bbox"],
                            "is_valid_plate": plate_result["is_valid_plate"],
                            "char_count": len(plate_result["plate_text"]),
                            "processing_time": time.time() - start_time
                        }
                        detections.append(detection)

            # Generar imagen con detecciones si hay alguna
            frame_image_base64 = None
            if detections and processing_params.get("include_frame_preview", True):
                frame_image_base64 = self._encode_frame_with_detections(frame, detections)

            processing_time = time.time() - start_time

            return FrameProcessingResult(
                frame_num=frame_num,
                timestamp=time.time(),
                processing_time=processing_time,
                detections=detections,
                frame_image_base64=frame_image_base64,
                success=True
            )

        except Exception as e:
            logger.warning(f"⚠️ Error procesando frame {frame_num}: {str(e)}")
            return FrameProcessingResult(
                frame_num=frame_num,
                timestamp=time.time(),
                processing_time=time.time() - start_time,
                detections=[],
                success=False,
                error=str(e)
            )

    async def _send_streaming_update(
            self,
            session_id: str,
            frames_buffer: List[FrameProcessingResult],
            detection_tracker: 'DetectionTracker'
    ):
        """Envía actualización de streaming al cliente"""

        try:
            session = connection_manager.get_session(session_id)
            if not session:
                return

            # Preparar datos de actualización
            current_frame = frames_buffer[-1].frame_num if frames_buffer else session.current_frame
            progress_percent = (session.processed_frames / max(
                session.total_frames // session.processing_params.get("frame_skip", 2), 1)) * 100

            # Obtener detecciones del último frame con detecciones
            latest_detections = []
            latest_frame_image = None

            for frame_result in reversed(frames_buffer):
                if frame_result.detections:
                    latest_detections = frame_result.detections
                    latest_frame_image = frame_result.frame_image_base64
                    break

            # Datos de la actualización
            update_data = {
                "frame_number": current_frame,
                "processed_frames": session.processed_frames,
                "total_frames": session.total_frames,
                "progress_percent": round(progress_percent, 2),
                "processing_speed": round(session.processing_speed, 2),

                # Detecciones del frame actual
                "current_detections": latest_detections,
                "frame_image": latest_frame_image,

                # Resumen acumulado
                "detection_summary": {
                    "total_detections": len(session.detections),
                    "unique_plates_count": len(session.unique_plates),
                    "frames_with_detections": session.frames_with_detections,
                    "best_detection": session.best_detection
                },

                # Placas únicas actualizadas
                "unique_plates": list(session.unique_plates.values()),

                # Métricas de tiempo
                "timing": {
                    "elapsed_time": time.time() - session.start_time if session.start_time else 0,
                    "estimated_remaining": self._estimate_remaining_time(session)
                }
            }

            # Enviar actualización
            await connection_manager.broadcast_to_session(
                session_id, "processing_update", update_data
            )

        except Exception as e:
            logger.error(f"❌ Error enviando actualización de streaming: {str(e)}")

    def _estimate_remaining_time(self, session) -> float:
        """Estima tiempo restante de procesamiento"""
        if not session.start_time or session.processing_speed <= 0:
            return 0.0

        frame_skip = session.processing_params.get("frame_skip", 2)
        total_frames_to_process = session.total_frames // frame_skip
        remaining_frames = total_frames_to_process - session.processed_frames

        return remaining_frames / session.processing_speed

    async def pause_processing(self, session_id: str):
        """Pausa el procesamiento de una sesión"""
        logger.info(f"⏸️ Pausando procesamiento: {session_id}")
        # El control se maneja en el connection_manager

    async def resume_processing(self, session_id: str):
        """Reanuda el procesamiento de una sesión"""
        logger.info(f"▶️ Reanudando procesamiento: {session_id}")
        # El control se maneja en el connection_manager

    async def stop_processing(self, session_id: str):
        """Detiene el procesamiento de una sesión"""
        logger.info(f"⏹️ Deteniendo procesamiento: {session_id}")

        # Cancelar tarea de procesamiento si existe
        if session_id in self.active_processing_tasks:
            task = self.active_processing_tasks[session_id]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self.active_processing_tasks[session_id]

    def cleanup(self):
        """Limpia recursos del servicio"""
        if self.executor:
            self.executor.shutdown(wait=False)

        # Cancelar todas las tareas activas
        for task in self.active_processing_tasks.values():
            if not task.done():
                task.cancel()

    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas globales de procesamiento"""

        try:
            active_sessions = len(self.active_processing_tasks)
            total_sessions = len(connection_manager.sessions)

            # Estadísticas de sesiones activas
            processing_stats = {
                "active_processing_sessions": active_sessions,
                "total_sessions": total_sessions,
                "processing_tasks": list(self.active_processing_tasks.keys())
            }

            # Estadísticas de rendimiento por sesión
            session_stats = {}
            for session_id, session in connection_manager.sessions.items():
                if session.start_time:
                    session_stats[session_id] = {
                        "status": session.status.value,
                        "progress_percent": (session.processed_frames / max(session.total_frames, 1)) * 100,
                        "processing_speed": session.processing_speed,
                        "detections_count": len(session.detections),
                        "uptime": time.time() - session.start_time
                    }

            processing_stats["sessions"] = session_stats

            # Estadísticas del thread pool
            if self.executor:
                processing_stats["executor"] = {
                    "max_workers": self.executor._max_workers,
                    "threads": len(self.executor._threads) if hasattr(self.executor, '_threads') else 0
                }

            return processing_stats

        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas de procesamiento: {str(e)}")
            return {"error": str(e)}

    async def optimize_performance_settings(self, session_id: str) -> Dict[str, Any]:
        """Optimiza configuraciones de rendimiento para una sesión"""

        try:
            session = connection_manager.get_session(session_id)
            if not session:
                return {"error": "Sesión no encontrada"}

            # Analizar rendimiento actual
            current_speed = session.processing_speed
            target_speed = 5.0  # frames por segundo objetivo

            # Sugerir optimizaciones
            suggestions = []

            if current_speed < target_speed:
                current_frame_skip = session.processing_params.get("frame_skip", 2)

                if current_frame_skip < 5:
                    suggestions.append({
                        "parameter": "frame_skip",
                        "current": current_frame_skip,
                        "suggested": min(current_frame_skip + 1, 5),
                        "reason": "Aumentar frame_skip para mayor velocidad"
                    })

                current_confidence = session.processing_params.get("confidence_threshold", 0.3)
                if current_confidence < 0.4:
                    suggestions.append({
                        "parameter": "confidence_threshold",
                        "current": current_confidence,
                        "suggested": min(current_confidence + 0.1, 0.5),
                        "reason": "Aumentar umbral de confianza para menos detecciones"
                    })

            elif current_speed > target_speed * 2:
                # Sistema muy rápido, puede mejorar calidad
                current_frame_skip = session.processing_params.get("frame_skip", 2)

                if current_frame_skip > 1:
                    suggestions.append({
                        "parameter": "frame_skip",
                        "current": current_frame_skip,
                        "suggested": max(current_frame_skip - 1, 1),
                        "reason": "Reducir frame_skip para mayor precisión"
                    })

            return {
                "session_id": session_id,
                "current_performance": {
                    "processing_speed": current_speed,
                    "target_speed": target_speed,
                    "performance_ratio": current_speed / target_speed
                },
                "suggestions": suggestions,
                "status": "optimal" if not suggestions else "can_improve"
            }

        except Exception as e:
            logger.error(f"❌ Error optimizando rendimiento: {str(e)}")
            return {"error": str(e)}


class DetectionTracker:
    """Tracker de detecciones para streaming en tiempo real"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.unique_plates: Dict[str, Dict[str, Any]] = {}
        self.all_detections: List[Dict[str, Any]] = []
        self.best_detection: Optional[Dict[str, Any]] = None
        self.detection_history: List[Dict[str, Any]] = []  # Para análisis temporal

    def update_detections(self, detections: List[Dict[str, Any]], frame_num: int):
        """Actualiza tracker con nuevas detecciones"""

        for detection in detections:
            plate_text = detection["plate_text"]
            confidence = detection["overall_confidence"]

            # Agregar a lista completa
            self.all_detections.append(detection)

            # Agregar al historial con timestamp
            self.detection_history.append({
                **detection,
                "frame_num": frame_num,
                "detection_time": time.time()
            })

            # Actualizar placas únicas
            if plate_text not in self.unique_plates:
                self.unique_plates[plate_text] = {
                    "plate_text": plate_text,
                    "first_seen_frame": frame_num,
                    "last_seen_frame": frame_num,
                    "detection_count": 0,
                    "best_confidence": 0.0,
                    "best_frame": frame_num,
                    "best_detection": detection,
                    "is_valid_format": detection["is_valid_plate"],
                    "total_confidence": 0.0,
                    "avg_confidence": 0.0,
                    "timestamps": [],
                    "confidence_history": [],
                    "stability_score": 0.0
                }

            # Actualizar datos de la placa
            plate_data = self.unique_plates[plate_text]
            plate_data["detection_count"] += 1
            plate_data["last_seen_frame"] = frame_num
            plate_data["total_confidence"] += confidence
            plate_data["avg_confidence"] = plate_data["total_confidence"] / plate_data["detection_count"]
            plate_data["timestamps"].append(detection["timestamp"])
            plate_data["confidence_history"].append(confidence)

            # Calcular score de estabilidad
            if len(plate_data["confidence_history"]) > 1:
                confidences = plate_data["confidence_history"]
                std_dev = np.std(confidences)
                mean_conf = np.mean(confidences)
                plate_data["stability_score"] = 1.0 - min(std_dev / max(mean_conf, 0.1), 1.0)

            # Actualizar mejor detección de esta placa
            if confidence > plate_data["best_confidence"]:
                plate_data["best_confidence"] = confidence
                plate_data["best_frame"] = frame_num
                plate_data["best_detection"] = detection

            # Actualizar mejor detección global
            if self.best_detection is None or confidence > self.best_detection["overall_confidence"]:
                self.best_detection = detection

    def get_unique_plates(self) -> Dict[str, Dict[str, Any]]:
        """Retorna diccionario de placas únicas"""
        return self.unique_plates

    def get_best_detection(self) -> Optional[Dict[str, Any]]:
        """Retorna la mejor detección global"""
        return self.best_detection

    def get_detection_timeline(self) -> List[Dict[str, Any]]:
        """Retorna timeline de detecciones para análisis temporal"""
        return sorted(self.detection_history, key=lambda x: x["frame_num"])

    def get_plates_by_quality(self) -> Dict[str, List[Dict[str, Any]]]:
        """Clasifica placas por calidad de detección"""

        quality_groups = {
            "excellent": [],  # confidence > 0.8 y stability > 0.7
            "good": [],  # confidence > 0.6 y stability > 0.5
            "fair": [],  # confidence > 0.4 y stability > 0.3
            "poor": []  # resto
        }

        for plate_data in self.unique_plates.values():
            confidence = plate_data["best_confidence"]
            stability = plate_data["stability_score"]

            if confidence > 0.8 and stability > 0.7:
                quality_groups["excellent"].append(plate_data)
            elif confidence > 0.6 and stability > 0.5:
                quality_groups["good"].append(plate_data)
            elif confidence > 0.4 and stability > 0.3:
                quality_groups["fair"].append(plate_data)
            else:
                quality_groups["poor"].append(plate_data)

        return quality_groups

    def get_final_summary(self) -> Dict[str, Any]:
        """Genera resumen final del tracking"""

        # Ordenar placas por mejor confianza
        sorted_plates = sorted(
            self.unique_plates.values(),
            key=lambda p: p["best_confidence"],
            reverse=True
        )

        # Estadísticas
        valid_plates = [p for p in sorted_plates if p["is_valid_format"]]
        total_detections = len(self.all_detections)

        # Análisis de calidad
        quality_groups = self.get_plates_by_quality()

        # Estadísticas temporales
        timeline = self.get_detection_timeline()
        detection_frames = list(set(d["frame_num"] for d in timeline))

        # Análisis de confianza
        if self.all_detections:
            confidences = [d["overall_confidence"] for d in self.all_detections]
            confidence_stats = {
                "avg": np.mean(confidences),
                "max": np.max(confidences),
                "min": np.min(confidences),
                "std": np.std(confidences),
                "median": np.median(confidences)
            }
        else:
            confidence_stats = {
                "avg": 0, "max": 0, "min": 0, "std": 0, "median": 0
            }

        return {
            "total_detections": total_detections,
            "unique_plates_count": len(self.unique_plates),
            "valid_plates_count": len(valid_plates),
            "unique_plates": sorted_plates,
            "best_plate": sorted_plates[0] if sorted_plates else None,

            "quality_analysis": {
                "excellent_count": len(quality_groups["excellent"]),
                "good_count": len(quality_groups["good"]),
                "fair_count": len(quality_groups["fair"]),
                "poor_count": len(quality_groups["poor"]),
                "quality_groups": quality_groups
            },

            "detection_rate": {
                "frames_with_detections": len(detection_frames),
                "avg_detections_per_frame": total_detections / max(len(detection_frames), 1),
                "detection_density": len(detection_frames) / max(
                    max(d["frame_num"] for d in timeline) if timeline else 1, 1
                )
            },

            "confidence_statistics": confidence_stats,

            "temporal_analysis": {
                "first_detection_frame": min(d["frame_num"] for d in timeline) if timeline else 0,
                "last_detection_frame": max(d["frame_num"] for d in timeline) if timeline else 0,
                "detection_span_frames": (
                    max(d["frame_num"] for d in timeline) - min(d["frame_num"] for d in timeline)
                    if len(timeline) > 1 else 0
                )
            },

            "session_info": {
                "session_id": self.session_id,
                "processing_completed": True,
                "total_unique_plates": len(self.unique_plates),
                "has_valid_detections": len(valid_plates) > 0
            }
        }


# Instancia global del servicio
real_time_video_service = RealTimeVideoService()


async def _send_streaming_update(
        self,
        session_id: str,
        frames_buffer: List[FrameProcessingResult],
        detection_tracker: 'DetectionTracker'
):
    """Envía actualización de streaming al cliente"""

    try:
        session = connection_manager.get_session(session_id)
        if not session:
            return

        # Preparar datos de actualización
        current_frame = frames_buffer[-1].frame_num if frames_buffer else session.current_frame
        progress_percent = (session.processed_frames / max(
            session.total_frames // session.processing_params.get("frame_skip", 2), 1)) * 100

        # Obtener detecciones del último frame con detecciones
        latest_detections = []
        latest_frame_image = None

        for frame_result in reversed(frames_buffer):
            if frame_result.detections:
                latest_detections = frame_result.detections
                latest_frame_image = frame_result.frame_image_base64
                break

        # Datos de la actualización
        update_data = {
            "frame_number": current_frame,
            "processed_frames": session.processed_frames,
            "total_frames": session.total_frames,
            "progress_percent": round(progress_percent, 2),
            "processing_speed": round(session.processing_speed, 2),

            # Detecciones del frame actual
            "current_detections": latest_detections,
            "frame_image": latest_frame_image,

            # Resumen acumulado
            "detection_summary": {
                "total_detections": len(session.detections),
                "unique_plates_count": len(session.unique_plates),
                "frames_with_detections": session.frames_with_detections,
                "best_detection": session.best_detection
            },

            # Placas únicas actualizadas
            "unique_plates": list(session.unique_plates.values()),

            # Métricas de tiempo
            "timing": {
                "elapsed_time": time.time() - session.start_time if session.start_time else 0,
                "estimated_remaining": self._estimate_remaining_time(session)
            }
        }

        # Enviar actualización
        await connection_manager.broadcast_to_session(
            session_id, "processing_update", update_data
        )

    except Exception as e:
        logger.error(f"❌ Error enviando actualización de streaming: {str(e)}")


def _estimate_remaining_time(self, session) -> float:
    """Estima tiempo restante de procesamiento"""
    if not session.start_time or session.processing_speed <= 0:
        return 0.0

    frame_skip = session.processing_params.get("frame_skip", 2)
    total_frames_to_process = session.total_frames // frame_skip
    remaining_frames = total_frames_to_process - session.processed_frames

    return remaining_frames / session.processing_speed


async def pause_processing(self, session_id: str):
    """Pausa el procesamiento de una sesión"""
    logger.info(f"⏸️ Pausando procesamiento: {session_id}")
    # El control se maneja en el connection_manager


async def resume_processing(self, session_id: str):
    """Reanuda el procesamiento de una sesión"""
    logger.info(f"▶️ Reanudando procesamiento: {session_id}")
    # El control se maneja en el connection_manager


async def stop_processing(self, session_id: str):
    """Detiene el procesamiento de una sesión"""
    logger.info(f"⏹️ Deteniendo procesamiento: {session_id}")

    # Cancelar tarea de procesamiento si existe
    if session_id in self.active_processing_tasks:
        task = self.active_processing_tasks[session_id]
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        del self.active_processing_tasks[session_id]


def cleanup(self):
    """Limpia recursos del servicio"""
    if self.executor:
        self.executor.shutdown(wait=False)

    # Cancelar todas las tareas activas
    for task in self.active_processing_tasks.values():
        if not task.done():
            task.cancel()


class DetectionTracker:
    """Tracker de detecciones para streaming en tiempo real"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.unique_plates: Dict[str, Dict[str, Any]] = {}
        self.all_detections: List[Dict[str, Any]] = []
        self.best_detection: Optional[Dict[str, Any]] = None

    def update_detections(self, detections: List[Dict[str, Any]], frame_num: int):
        """Actualiza tracker con nuevas detecciones"""

        for detection in detections:
            plate_text = detection["plate_text"]
            confidence = detection["overall_confidence"]

            # Agregar a lista completa
            self.all_detections.append(detection)

            # Actualizar placas únicas
            if plate_text not in self.unique_plates:
                self.unique_plates[plate_text] = {
                    "plate_text": plate_text,
                    "first_seen_frame": frame_num,
                    "last_seen_frame": frame_num,
                    "detection_count": 0,
                    "best_confidence": 0.0,
                    "best_frame": frame_num,
                    "best_detection": detection,
                    "is_valid_format": detection["is_valid_plate"],
                    "total_confidence": 0.0,
                    "avg_confidence": 0.0,
                    "timestamps": []
                }

            # Actualizar datos de la placa
            plate_data = self.unique_plates[plate_text]
            plate_data["detection_count"] += 1
            plate_data["last_seen_frame"] = frame_num
            plate_data["total_confidence"] += confidence
            plate_data["avg_confidence"] = plate_data["total_confidence"] / plate_data["detection_count"]
            plate_data["timestamps"].append(detection["timestamp"])

            # Actualizar mejor detección de esta placa
            if confidence > plate_data["best_confidence"]:
                plate_data["best_confidence"] = confidence
                plate_data["best_frame"] = frame_num
                plate_data["best_detection"] = detection

            # Actualizar mejor detección global
            if self.best_detection is None or confidence > self.best_detection["overall_confidence"]:
                self.best_detection = detection

    def get_unique_plates(self) -> Dict[str, Dict[str, Any]]:
        """Retorna diccionario de placas únicas"""
        return self.unique_plates

    def get_best_detection(self) -> Optional[Dict[str, Any]]:
        """Retorna la mejor detección global"""
        return self.best_detection

    def get_final_summary(self) -> Dict[str, Any]:
        """Genera resumen final del tracking"""

        # Ordenar placas por mejor confianza
        sorted_plates = sorted(
            self.unique_plates.values(),
            key=lambda p: p["best_confidence"],
            reverse=True
        )

        # Estadísticas
        valid_plates = [p for p in sorted_plates if p["is_valid_format"]]
        total_detections = len(self.all_detections)

        return {
            "total_detections": total_detections,
            "unique_plates_count": len(self.unique_plates),
            "valid_plates_count": len(valid_plates),
            "unique_plates": sorted_plates,
            "best_plate": sorted_plates[0] if sorted_plates else None,
            "detection_rate": {
                "frames_with_detections": len(set(d["frame_num"] for d in self.all_detections)),
                "avg_detections_per_frame": total_detections / max(
                    len(set(d["frame_num"] for d in self.all_detections)), 1)
            },
            "confidence_stats": {
                "avg_confidence": sum(d["overall_confidence"] for d in self.all_detections) / max(total_detections, 1),
                "max_confidence": max((d["overall_confidence"] for d in self.all_detections), default=0.0),
                "min_confidence": min((d["overall_confidence"] for d in self.all_detections), default=0.0)
            }
        }


# Instancia global del servicio
real_time_video_service = RealTimeVideoService()


def _encode_frame_with_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> str:
    """
    Codifica frame con detecciones dibujadas a base64

    Args:
        frame: Frame original en RGB
        detections: Lista de detecciones

    Returns:
        Frame codificado en base64
    """
    try:
        frame_with_boxes = frame.copy()

        # Dibujar cada detección
        for detection in detections:
            bbox = detection["plate_bbox"]
            plate_text = detection["plate_text"]
            confidence = detection["overall_confidence"]
            is_valid = detection["is_valid_plate"]

            # Coordenadas del rectángulo
            x1, y1, x2, y2 = map(int, bbox)

            # Color basado en validez y confianza
            if is_valid:
                color = (0, 255, 0)  # Verde para placas válidas
            elif confidence > 0.5:
                color = (255, 255, 0)  # Amarillo para detecciones con buena confianza
            else:
                color = (255, 165, 0)  # Naranja para detecciones con baja confianza

            # Dibujar rectángulo
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)

            # Preparar etiqueta
            label = f"{plate_text} ({confidence:.2f})"
            if is_valid:
                label += " ✓"

            # Calcular tamaño del texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Fondo para el texto
            cv2.rectangle(
                frame_with_boxes,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Texto
            cv2.putText(
                frame_with_boxes,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),  # Texto blanco
                thickness
            )

        # Redimensionar para streaming (optimización de ancho de banda)
        config = self.processing_config["frame_processing"]
        max_size = config["max_size"]

        height, width = frame_with_boxes.shape[:2]
        if width > max_size:
            scale = max_size / width
            new_width = max_size
            new_height = int(height * scale)
            frame_with_boxes = cv2.resize(frame_with_boxes, (new_width, new_height))

        # Convertir a BGR para OpenCV
        frame_bgr = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)

        # Comprimir a JPEG con calidad configurada
        quality = config["quality"]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)

        if success:
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return frame_base64
        else:
            logger.warning("⚠️ Error codificando frame a JPEG")
            return ""

    except Exception as e:
        logger.warning(f"⚠️ Error codificando frame con detecciones: {str(e)}")
        return ""


class RealTimeVideoService:
    """Servicio principal para procesamiento de video en tiempo real"""

    def __init__(self):
        self.model_manager = model_manager
        self.file_service = file_service
        self.frame_processor = RealTimeFrameProcessor()

        # Thread pool para procesamiento de frames
        self.executor = ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="realtime_processor"
        )

        # Control de sesiones activas
        self.active_processing_tasks: Dict[str, asyncio.Task] = {}

        logger.info("🎬 RealTimeVideoService inicializado")

    async def process_video_streaming(
            self,
            video_path: str,
            file_info: Dict[str, Any],
            processing_params: Dict[str, Any]
    ):
        """
        Procesa un video completo con streaming en tiempo real vía WebSocket

        Args:
            video_path: Ruta del archivo de video
            file_info: Información del archivo
            processing_params: Parámetros de procesamiento
        """
        session_id = processing_params["session_id"]

        try:
            logger.info(f"🎬 Iniciando streaming para sesión: {session_id}")

            # Obtener sesión
            session = connection_manager.get_session(session_id)
            if not session:
                logger.error(f"❌ Sesión no encontrada: {session_id}")
                return

            # Almacenar tarea de procesamiento
            current_task = asyncio.current_task()
            self.active_processing_tasks[session_id] = current_task

            # Obtener información del video
            video_info = get_video_info(video_path)
            if not video_info:
                raise Exception("No se pudo obtener información del video")

            # Actualizar información en la sesión
            session.video_info = video_info
            session.total_frames = video_info["frame_count"]
            session.start_time = time.time()

            connection_manager.update_session_status(session_id, StreamingStatus.PROCESSING)

            # Enviar información inicial del video
            await connection_manager.broadcast_to_session(session_id, "processing_started", {
                "video_info": video_info,
                "total_frames": video_info["frame_count"],
                "fps": video_info["fps"],
                "duration_seconds": video_info["duration_seconds"],
                "file_info": file_info,
                "processing_params": processing_params
            })

            # Procesar video frame por frame
            await self._process_video_frames(session_id, video_path, processing_params)

            logger.success(f"✅ Streaming completado para sesión: {session_id}")

        except Exception as e:
            logger.error(f"❌ Error en streaming {session_id}: {str(e)}")

            # Actualizar estado de error
            if session_id in connection_manager.sessions:
                connection_manager.update_session_status(session_id, StreamingStatus.ERROR)
                await connection_manager.send_message(session_id, {
                    "type": "processing_error",
                    "error": str(e),
                    "timestamp": time.time()
                })
        finally:
            # Limpiar archivo temporal
            file_service.cleanup_temp_file(video_path)

            # Remover tarea de procesamiento
            if session_id in self.active_processing_tasks:
                del self.active_processing_tasks[session_id]

    async def _process_video_frames(
            self,
            session_id: str,
            video_path: str,
            processing_params: Dict[str, Any]
    ):
        """Procesa los frames del video con streaming en tiempo real"""

        session = connection_manager.get_session(session_id)
        if not session:
            return

        # Configuración de procesamiento
        frame_skip = processing_params.get("frame_skip", 2)
        max_duration = processing_params.get("max_duration", 600)

        # Control de flujo para streaming
        streaming_config = settings.get_streaming_config()
        send_interval = streaming_config["frame_processing"]["send_interval"]
        buffer_size = streaming_config["frame_processing"]["buffer_size"]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Variables de control
        frame_num = 0
        processed_count = 0
        frames_buffer = []
        last_send_time = time.time()
        detection_tracker = DetectionTracker(session_id)

        try:
            while True:
                # Verificar si la sesión debe detenerse
                if session.should_stop:
                    logger.info(f"🛑 Deteniendo procesamiento por solicitud: {session_id}")
                    break

                # Verificar si está pausada
                while session.is_paused and not session.should_stop:
                    await asyncio.sleep(0.1)

                # Leer frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Verificar límite de duración
                current_time_in_video = frame_num / fps
                if current_time_in_video > max_duration:
                    logger.info(f"⏰ Límite de duración alcanzado: {max_duration}s")
                    break

                # Procesar solo cada N frames
                if frame_num % frame_skip == 0:
                    try:
                        # Convertir frame a RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Procesar frame de forma asíncrona
                        frame_result = await self._process_frame_async(
                            frame_rgb, frame_num, fps, processing_params
                        )

                        processed_count += 1
                        session.processed_frames = processed_count
                        session.current_frame = frame_num

                        # Actualizar velocidad de procesamiento
                        if session.start_time:
                            elapsed = time.time() - session.start_time
                            session.processing_speed = processed_count / max(elapsed, 1)

                        # Agregar a buffer
                        frames_buffer.append(frame_result)

                        # Procesar detecciones
                        if frame_result.detections:
                            session.frames_with_detections += 1
                            session.total_detection_count += len(frame_result.detections)

                            # Actualizar tracker de detecciones
                            detection_tracker.update_detections(frame_result.detections, frame_num)

                            # Actualizar sesión con detecciones
                            session.detections.extend(frame_result.detections)
                            session.unique_plates = detection_tracker.get_unique_plates()

                            # Actualizar mejor detección
                            best_detection = detection_tracker.get_best_detection()
                            if best_detection:
                                session.best_detection = best_detection

                        # Enviar datos si es momento adecuado
                        current_time = time.time()
                        should_send = (
                                current_time - last_send_time >= send_interval or
                                len(frames_buffer) >= buffer_size or
                                len(frame_result.detections) > 0 or
                                processed_count % 30 == 0  # Cada 30 frames procesados para progreso
                        )

                        if should_send:
                            await self._send_streaming_update(
                                session_id, frames_buffer, detection_tracker
                            )
                            frames_buffer = []
                            last_send_time = current_time

                        # Pequeña pausa para no saturar
                        await asyncio.sleep(0.01)

                    except Exception as e:
                        logger.warning(f"⚠️ Error procesando frame {frame_num}: {str(e)}")

                frame_num += 1

            # Enviar último lote de datos
            if frames_buffer:
                await self._send_streaming_update(session_id, frames_buffer, detection_tracker)

            # Procesamiento completado
            session.status = StreamingStatus.COMPLETED
            final_summary = detection_tracker.get_final_summary()

            await connection_manager.send_message(session_id, {
                "type": "processing_completed",
                "data": final_summary,
                "timestamp": time.time()
            })

        finally:
            cap.release()

    async def _process_frame_async(
            self,
            frame: np.ndarray,
            frame_num: int,
            fps: float,
            processing_params: Dict[str, Any]
    ) -> FrameProcessingResult:
        """Procesa un frame de forma asíncrona"""

        try:
            # Ejecutar procesamiento en thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.frame_processor.process_frame_sync,
                frame, frame_num, processing_params
            )

            # Corregir timestamp con FPS real
            result.timestamp = frame_num / fps

            return result

        except Exception as e:
            logger.warning(f"⚠️ Error en procesamiento asíncrono frame {frame_num}: {str(e)}")
            return FrameProcessingResult(
                frame_num=frame_num,
                timestamp=frame_num / fps,
                processing_time=0.0,
                detections=[],
                success=False,
                error=