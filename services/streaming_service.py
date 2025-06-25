"""
Servicio completo de streaming para procesamiento de video ALPR en tiempo real
Integra procesamiento de frames, tracking avanzado y comunicación WebSocket
"""

import cv2
import numpy as np
import time
import asyncio
import base64
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque

from config.settings import settings
from models.model_manager import model_manager
from services.file_service import file_service
from api.websocket_manager import connection_manager, StreamingStatus
from core.utils import PerformanceTimer, get_video_info, format_duration


@dataclass
class StreamingFrame:
    """Representa un frame procesado para streaming"""
    frame_num: int
    timestamp: float
    processing_time: float
    detections: List[Dict[str, Any]]
    frame_image_base64: Optional[str] = None
    frame_small_base64: Optional[str] = None  # Thumbnail
    success: bool = True
    error: Optional[str] = None

    # Metadatos adicionales
    original_size: Optional[Tuple[int, int]] = None
    compressed_size: Optional[int] = None
    quality_used: Optional[int] = None


class AdaptiveQualityManager:
    """Gestor de calidad adaptativa para optimizar ancho de banda"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.quality_history = deque(maxlen=10)
        self.latency_history = deque(maxlen=5)
        self.current_quality = settings.streaming_frame_quality
        self.target_fps = 2.0  # FPS objetivo para streaming

    def adjust_quality(self, processing_time: float, frame_size: int, connection_speed: float = 1.0) -> int:
        """
        Ajusta la calidad basándose en métricas de rendimiento

        Args:
            processing_time: Tiempo de procesamiento del frame
            frame_size: Tamaño del frame en bytes
            connection_speed: Velocidad estimada de conexión (0.0-1.0)

        Returns:
            Nueva calidad recomendada (10-100)
        """
        # Agregar métricas al historial
        self.quality_history.append({
            "quality": self.current_quality,
            "processing_time": processing_time,
            "frame_size": frame_size,
            "timestamp": time.time()
        })

        # Si el procesamiento es muy lento, reducir calidad
        if processing_time > 1.0:  # Más de 1 segundo por frame
            self.current_quality = max(30, self.current_quality - 15)
        elif processing_time > 0.5:  # Más de 0.5 segundos
            self.current_quality = max(40, self.current_quality - 10)
        elif processing_time < 0.2:  # Muy rápido, puede mejorar calidad
            self.current_quality = min(85, self.current_quality + 5)

        # Ajustar por tamaño de frame
        if frame_size > 200000:  # Más de 200KB
            self.current_quality = max(25, self.current_quality - 10)
        elif frame_size < 50000:  # Menos de 50KB
            self.current_quality = min(90, self.current_quality + 5)

        # Ajustar por velocidad de conexión estimada
        if connection_speed < 0.3:  # Conexión lenta
            self.current_quality = max(20, self.current_quality - 20)
        elif connection_speed > 0.8:  # Conexión rápida
            self.current_quality = min(95, self.current_quality + 10)

        # Límites finales
        self.current_quality = max(15, min(95, self.current_quality))

        return self.current_quality

    def get_recommended_frame_skip(self) -> int:
        """Recomienda frame_skip basado en rendimiento"""
        if len(self.quality_history) < 3:
            return 2

        avg_processing_time = sum(h["processing_time"] for h in self.quality_history) / len(self.quality_history)

        if avg_processing_time > 1.0:
            return 5  # Muy lento, saltar más frames
        elif avg_processing_time > 0.5:
            return 3  # Lento, saltar algunos frames
        else:
            return 2  # Normal, frame skip mínimo


class StreamingDetectionTracker:
    """Tracker especializado para streaming que mantiene historial temporal"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.unique_plates: Dict[str, Dict[str, Any]] = {}
        self.detection_timeline: List[Dict[str, Any]] = []
        self.frame_detections: Dict[int, List[Dict[str, Any]]] = {}
        self.best_detection_per_plate: Dict[str, Dict[str, Any]] = {}

    def add_frame_detections(self, frame_num: int, detections: List[Dict[str, Any]], timestamp: float):
        """Agrega detecciones de un frame específico"""
        # Guardar detecciones por frame
        self.frame_detections[frame_num] = detections

        for detection in detections:
            plate_text = detection["plate_text"]
            confidence = detection["overall_confidence"]

            # Agregar al timeline
            timeline_entry = {
                **detection,
                "frame_num": frame_num,
                "timestamp": timestamp,
                "detection_id": f"{frame_num}_{plate_text}_{int(time.time() * 1000)}"
            }
            self.detection_timeline.append(timeline_entry)

            # Actualizar o crear entrada de placa única
            if plate_text not in self.unique_plates:
                self.unique_plates[plate_text] = {
                    "plate_text": plate_text,
                    "first_seen_frame": frame_num,
                    "first_seen_timestamp": timestamp,
                    "last_seen_frame": frame_num,
                    "last_seen_timestamp": timestamp,
                    "detection_count": 0,
                    "best_confidence": 0.0,
                    "best_frame": frame_num,
                    "best_timestamp": timestamp,
                    "avg_confidence": 0.0,
                    "total_confidence": 0.0,
                    "is_valid_format": detection.get("is_valid_plate", False),
                    "frame_history": [],
                    "confidence_trend": [],
                    "status": "active"
                }

            # Actualizar estadísticas
            plate_data = self.unique_plates[plate_text]
            plate_data["detection_count"] += 1
            plate_data["last_seen_frame"] = frame_num
            plate_data["last_seen_timestamp"] = timestamp
            plate_data["total_confidence"] += confidence
            plate_data["avg_confidence"] = plate_data["total_confidence"] / plate_data["detection_count"]

            # Actualizar mejor detección
            if confidence > plate_data["best_confidence"]:
                plate_data["best_confidence"] = confidence
                plate_data["best_frame"] = frame_num
                plate_data["best_timestamp"] = timestamp
                self.best_detection_per_plate[plate_text] = detection

            # Agregar a historial
            plate_data["frame_history"].append(frame_num)
            plate_data["confidence_trend"].append(confidence)

            # Mantener solo últimos 20 frames en historial
            if len(plate_data["frame_history"]) > 20:
                plate_data["frame_history"] = plate_data["frame_history"][-20:]
                plate_data["confidence_trend"] = plate_data["confidence_trend"][-20:]

    def get_detections_for_frame(self, frame_num: int) -> List[Dict[str, Any]]:
        """Obtiene las detecciones para un frame específico"""
        return self.frame_detections.get(frame_num, [])

    def get_active_plates_at_frame(self, frame_num: int, tolerance: int = 30) -> List[Dict[str, Any]]:
        """Obtiene placas que estaban activas en un frame dado"""
        active_plates = []

        for plate_text, plate_data in self.unique_plates.items():
            # Verificar si la placa estaba activa en el rango de frames
            if (plate_data["first_seen_frame"] <= frame_num + tolerance and
                    plate_data["last_seen_frame"] >= frame_num - tolerance):
                active_plates.append(plate_data)

        return sorted(active_plates, key=lambda x: x["best_confidence"], reverse=True)

    def get_streaming_summary(self) -> Dict[str, Any]:
        """Genera resumen optimizado para streaming"""
        # Ordenar placas por confianza
        sorted_plates = sorted(
            self.unique_plates.values(),
            key=lambda p: p["best_confidence"],
            reverse=True
        )

        # Calcular estadísticas rápidas
        total_detections = len(self.detection_timeline)
        valid_plates = [p for p in sorted_plates if p["is_valid_format"]]
        frames_with_detections = len(self.frame_detections)

        return {
            "total_detections": total_detections,
            "unique_plates_count": len(self.unique_plates),
            "valid_plates_count": len(valid_plates),
            "frames_with_detections": frames_with_detections,
            "best_plates": sorted_plates[:5],  # Top 5 placas
            "latest_detections": self.detection_timeline[-10:],  # Últimas 10 detecciones
            "detection_density": frames_with_detections / max(len(self.frame_detections), 1),
            "session_id": self.session_id
        }


class StreamingVideoProcessor:
    """Procesador principal de video para streaming en tiempo real"""

    def __init__(self):
        self.model_manager = model_manager
        self.file_service = file_service
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.quality_managers: Dict[str, AdaptiveQualityManager] = {}
        self.detection_trackers: Dict[str, StreamingDetectionTracker] = {}

        # Thread pool para procesamiento paralelo
        self.executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="streaming_processor"
        )

        # Configuración de streaming
        self.streaming_config = settings.get_streaming_config()

        logger.info("🎬 StreamingVideoProcessor inicializado")

    async def start_video_streaming(
            self,
            session_id: str,
            video_path: str,
            file_info: Dict[str, Any],
            processing_params: Dict[str, Any]
    ) -> bool:
        """
        Inicia el streaming de procesamiento de video

        Args:
            session_id: ID de la sesión WebSocket
            video_path: Ruta del archivo de video
            file_info: Información del archivo
            processing_params: Parámetros de procesamiento

        Returns:
            True si se inició correctamente
        """
        try:
            logger.info(f"🎬 Iniciando streaming para sesión: {session_id}")

            # Verificar que la sesión existe
            session = connection_manager.get_session(session_id)
            if not session:
                logger.error(f"❌ Sesión no encontrada: {session_id}")
                return False

            # Obtener información del video
            video_info = get_video_info(video_path)
            if not video_info:
                raise Exception("No se pudo obtener información del video")

            # Crear gestores para esta sesión
            self.quality_managers[session_id] = AdaptiveQualityManager(session_id)
            self.detection_trackers[session_id] = StreamingDetectionTracker(session_id)

            # Configurar sesión
            session.video_path = video_path
            session.file_info = file_info
            session.video_info = video_info
            session.processing_params = processing_params
            session.total_frames = video_info["frame_count"]
            session.start_time = time.time()

            # Actualizar estado
            connection_manager.update_session_status(session_id, StreamingStatus.PROCESSING)

            # Enviar información inicial
            await connection_manager.send_message(session_id, {
                "type": "streaming_started",
                "data": {
                    "video_info": video_info,
                    "file_info": file_info,
                    "processing_params": processing_params,
                    "streaming_config": self.streaming_config,
                    "estimated_duration": self._estimate_processing_time(video_info, processing_params)
                }
            })

            # Iniciar procesamiento en background
            asyncio.create_task(self._process_video_stream(session_id))

            return True

        except Exception as e:
            logger.error(f"❌ Error iniciando streaming {session_id}: {str(e)}")
            await self._handle_streaming_error(session_id, str(e))
            return False

    async def _process_video_stream(self, session_id: str):
        """Procesa el video frame por frame con streaming"""
        session = connection_manager.get_session(session_id)
        if not session:
            return

        try:
            video_path = session.video_path
            processing_params = session.processing_params
            quality_manager = self.quality_managers[session_id]
            detection_tracker = self.detection_trackers[session_id]

            # Configuración de procesamiento
            frame_skip = processing_params.get("frame_skip", 2)
            max_duration = processing_params.get("max_duration", 600)
            confidence_threshold = processing_params.get("confidence_threshold", 0.3)

            # Abrir video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("No se pudo abrir el video")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Variables de control
            frame_num = 0
            processed_count = 0
            last_send_time = time.time()
            send_interval = self.streaming_config["frame_processing"]["send_interval"]

            logger.info(f"📹 Procesando video: {total_frames} frames a {fps} FPS")

            while True:
                # Verificar controles de sesión
                if session.should_stop:
                    logger.info(f"🛑 Deteniendo por solicitud: {session_id}")
                    break

                # Manejar pausa
                while session.is_paused and not session.should_stop:
                    await asyncio.sleep(0.1)
                    continue

                # Leer frame
                ret, frame = cap.read()
                if not ret:
                    logger.info(f"📹 Fin del video alcanzado: {session_id}")
                    break

                # Verificar límite de duración
                current_time_in_video = frame_num / fps
                if current_time_in_video > max_duration:
                    logger.info(f"⏰ Límite de duración alcanzado: {max_duration}s")
                    break

                # Procesar solo cada N frames (según frame_skip)
                if frame_num % frame_skip == 0:
                    try:
                        # Convertir a RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Procesar frame
                        streaming_frame = await self._process_single_frame(
                            session_id, frame_rgb, frame_num, fps, processing_params
                        )

                        processed_count += 1
                        session.processed_frames = processed_count
                        session.current_frame = frame_num

                        # Actualizar velocidad de procesamiento
                        if session.start_time:
                            elapsed = time.time() - session.start_time
                            session.processing_speed = processed_count / max(elapsed, 1)

                        # Actualizar tracker de detecciones
                        if streaming_frame.detections:
                            session.frames_with_detections += 1
                            session.total_detection_count += len(streaming_frame.detections)

                            detection_tracker.add_frame_detections(
                                frame_num, streaming_frame.detections, streaming_frame.timestamp
                            )

                            # Actualizar placas únicas en la sesión
                            session.unique_plates = detection_tracker.unique_plates

                            # Actualizar mejor detección
                            if detection_tracker.best_detection_per_plate:
                                best_overall = max(
                                    detection_tracker.best_detection_per_plate.values(),
                                    key=lambda x: x["overall_confidence"]
                                )
                                session.best_detection = best_overall

                        # Enviar datos si es momento adecuado
                        current_time = time.time()
                        should_send = (
                                current_time - last_send_time >= send_interval or
                                len(streaming_frame.detections) > 0 or
                                processed_count % 15 == 0  # Cada 15 frames para progreso
                        )

                        if should_send:
                            await self._send_streaming_update(session_id, streaming_frame, detection_tracker)
                            last_send_time = current_time

                        # Pequeña pausa para no saturar
                        await asyncio.sleep(0.02)

                    except Exception as e:
                        logger.warning(f"⚠️ Error procesando frame {frame_num}: {str(e)}")

                frame_num += 1

                # Actualizar progreso general
                if frame_num % 100 == 0:  # Cada 100 frames
                    progress = (frame_num / total_frames) * 100
                    logger.info(f"📊 Progreso {session_id}: {progress:.1f}% - "
                                f"Placas únicas: {len(detection_tracker.unique_plates)}")

            # Finalizar streaming
            cap.release()
            await self._finalize_streaming(session_id, detection_tracker)

        except Exception as e:
            logger.error(f"❌ Error en streaming {session_id}: {str(e)}")
            await self._handle_streaming_error(session_id, str(e))
        finally:
            # Limpiar recursos
            if session_id in self.quality_managers:
                del self.quality_managers[session_id]
            if session_id in self.detection_trackers:
                del self.detection_trackers[session_id]

            # Limpiar archivo temporal
            if session and session.video_path:
                self.file_service.cleanup_temp_file(session.video_path)

    async def _process_single_frame(
            self,
            session_id: str,
            frame: np.ndarray,
            frame_num: int,
            fps: float,
            processing_params: Dict[str, Any]
    ) -> StreamingFrame:
        """Procesa un frame individual optimizado para streaming"""

        start_time = time.time()
        quality_manager = self.quality_managers[session_id]

        try:
            # Parámetros de procesamiento
            confidence_threshold = processing_params.get("confidence_threshold", 0.3)
            iou_threshold = processing_params.get("iou_threshold", 0.4)

            # Procesar con el pipeline ALPR
            loop = asyncio.get_event_loop()
            pipeline_result = await loop.run_in_executor(
                self.executor,
                self._process_frame_sync,
                frame, confidence_threshold, iou_threshold
            )

            # Extraer detecciones
            detections = []
            if pipeline_result.get("success") and pipeline_result.get("final_results"):
                for i, plate_result in enumerate(pipeline_result["final_results"]):
                    if plate_result["plate_text"]:
                        detection = {
                            "detection_id": f"{session_id}_{frame_num}_{i}",
                            "frame_num": frame_num,
                            "timestamp": frame_num / fps,
                            "plate_text": plate_result["plate_text"],
                            "plate_confidence": plate_result["plate_confidence"],
                            "char_confidence": plate_result.get("character_recognition", {}).get("confidence", 0.0),
                            "overall_confidence": plate_result["overall_confidence"],
                            "plate_bbox": plate_result["plate_bbox"],
                            "is_valid_plate": plate_result["is_valid_plate"],
                            "char_count": len(plate_result["plate_text"]),
                            "bbox_area": self._calculate_bbox_area(plate_result["plate_bbox"])
                        }
                        detections.append(detection)

            # Generar imagen con detecciones para streaming
            frame_base64 = None
            frame_small_base64 = None

            if detections or processing_params.get("send_all_frames", False):
                # Crear frame con detecciones dibujadas
                annotated_frame = self._draw_detections_on_frame(frame, detections)

                # Codificar frame principal
                frame_base64, compressed_size, quality_used = self._encode_frame_adaptive(
                    annotated_frame, quality_manager
                )

                # Crear thumbnail pequeño
                frame_small_base64 = self._create_frame_thumbnail(annotated_frame)

                # Actualizar calidad adaptativa
                processing_time = time.time() - start_time
                quality_manager.adjust_quality(processing_time, compressed_size or 0)

            processing_time = time.time() - start_time

            return StreamingFrame(
                frame_num=frame_num,
                timestamp=frame_num / fps,
                processing_time=processing_time,
                detections=detections,
                frame_image_base64=frame_base64,
                frame_small_base64=frame_small_base64,
                original_size=(frame.shape[1], frame.shape[0]),
                compressed_size=compressed_size,
                quality_used=quality_used,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"⚠️ Error procesando frame {frame_num}: {str(e)}")

            return StreamingFrame(
                frame_num=frame_num,
                timestamp=frame_num / fps,
                processing_time=processing_time,
                detections=[],
                success=False,
                error=str(e)
            )

    def _process_frame_sync(self, frame: np.ndarray, confidence: float, iou: float) -> Dict[str, Any]:
        """Procesamiento síncrono del frame para usar en executor"""
        try:
            return self.model_manager.process_full_pipeline(
                frame,
                conf=confidence,
                iou=iou,
                verbose=False
            )
        except Exception as e:
            logger.error(f"❌ Error en pipeline sync: {str(e)}")
            return {"success": False, "error": str(e)}

    def _draw_detections_on_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Dibuja las detecciones sobre el frame"""
        annotated_frame = frame.copy()

        for detection in detections:
            bbox = detection["plate_bbox"]
            plate_text = detection["plate_text"]
            confidence = detection["overall_confidence"]
            is_valid = detection["is_valid_plate"]

            # Coordenadas
            x1, y1, x2, y2 = map(int, bbox)

            # Color según validez
            if is_valid:
                color = (0, 255, 0)  # Verde para placas válidas
                thickness = 3
            elif confidence > 0.5:
                color = (255, 255, 0)  # Amarillo para buena confianza
                thickness = 2
            else:
                color = (255, 165, 0)  # Naranja para baja confianza
                thickness = 2

            # Dibujar rectángulo
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Preparar etiqueta
            label = f"{plate_text}"
            confidence_label = f"{confidence:.2f}"
            if is_valid:
                label += " ✓"

            # Dibujar etiqueta principal
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            label_thickness = 2

            # Calcular tamaño del texto
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)

            # Fondo para la etiqueta
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_h - baseline - 8),
                (x1 + text_w + 8, y1),
                color,
                -1
            )

            # Texto principal
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 4, y1 - baseline - 4),
                font,
                font_scale,
                (255, 255, 255),
                label_thickness
            )

            # Confianza en esquina superior derecha del bbox
            conf_font_scale = 0.5
            cv2.putText(
                annotated_frame,
                confidence_label,
                (x2 - 50, y1 + 20),
                font,
                conf_font_scale,
                color,
                1
            )

        return annotated_frame

    def _encode_frame_adaptive(
            self,
            frame: np.ndarray,
            quality_manager: AdaptiveQualityManager
    ) -> Tuple[str, int, int]:
        """Codifica frame con calidad adaptativa"""
        try:
            # Redimensionar si es necesario
            config = self.streaming_config["frame_processing"]
            max_size = config["max_size"]

            height, width = frame.shape[:2]
            if width > max_size:
                scale = max_size / width
                new_width = max_size
                new_height = int(height * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame

            # Convertir a BGR para OpenCV
            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)

            # Obtener calidad actual
            quality = quality_manager.current_quality

            # Comprimir a JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)

            if success:
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                compressed_size = len(buffer)
                return frame_base64, compressed_size, quality
            else:
                logger.warning("⚠️ Error codificando frame")
                return "", 0, quality

        except Exception as e:
            logger.warning(f"⚠️ Error en codificación adaptativa: {str(e)}")
            return "", 0, 50

    def _create_frame_thumbnail(self, frame: np.ndarray, size: int = 160) -> str:
        """Crea un thumbnail pequeño del frame"""
        try:
            # Redimensionar a thumbnail
            height, width = frame.shape[:2]
            if width > height:
                new_width = size
                new_height = int((size * height) / width)
            else:
                new_height = size
                new_width = int((size * width) / height)

            thumbnail = cv2.resize(frame, (new_width, new_height))

            # Convertir y comprimir con baja calidad
            thumbnail_bgr = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
            success, buffer = cv2.imencode('.jpg', thumbnail_bgr, encode_param)

            if success:
                return base64.b64encode(buffer).decode('utf-8')
            else:
                return ""

        except Exception as e:
            logger.warning(f"⚠️ Error creando thumbnail: {str(e)}")
            return ""

    def _calculate_bbox_area(self, bbox: List[float]) -> float:
        """Calcula área del bounding box"""
        x1, y1, x2, y2 = bbox
        return abs(x2 - x1) * abs(y2 - y1)

    async def _send_streaming_update(
            self,
            session_id: str,
            streaming_frame: StreamingFrame,
            detection_tracker: StreamingDetectionTracker
    ):
        """Envía actualización de streaming al cliente"""
        try:
            session = connection_manager.get_session(session_id)
            if not session:
                return

            # Calcular progreso
            progress_percent = 0
            if session.total_frames > 0:
                frames_to_process = session.total_frames // session.processing_params.get("frame_skip", 2)
                progress_percent = (session.processed_frames / max(frames_to_process, 1)) * 100

            # Obtener resumen de detecciones
            detection_summary = detection_tracker.get_streaming_summary()

            # Preparar datos de actualización
            update_data = {
                "frame_info": {
                    "frame_number": streaming_frame.frame_num,
                    "timestamp": streaming_frame.timestamp,
                    "processing_time": streaming_frame.processing_time,
                    "success": streaming_frame.success
                },
                "progress": {
                    "processed_frames": session.processed_frames,
                    "total_frames": session.total_frames,
                    "progress_percent": round(progress_percent, 2),
                    "processing_speed": round(session.processing_speed, 2)
                },
                "current_detections": streaming_frame.detections,
                "detection_summary": detection_summary,
                "timing": {
                    "elapsed_time": time.time() - session.start_time if session.start_time else 0,
                    "estimated_remaining": self._estimate_remaining_time(session)
                }
            }

            # Incluir frame si hay detecciones o se solicita
            if streaming_frame.frame_image_base64:
                update_data["frame_data"] = {
                    "image_base64": streaming_frame.frame_image_base64,
                    "thumbnail_base64": streaming_frame.frame_small_base64,
                    "original_size": streaming_frame.original_size,
                    "compressed_size": streaming_frame.compressed_size,
                    "quality_used": streaming_frame.quality_used
                }

            # Incluir información de calidad adaptativa
            if session_id in self.quality_managers:
                quality_manager = self.quality_managers[session_id]
                update_data["quality_info"] = {
                    "current_quality": quality_manager.current_quality,
                    "recommended_frame_skip": quality_manager.get_recommended_frame_skip(),
                    "adaptive_enabled": self.streaming_config["frame_processing"]["adaptive_quality"]
                }

            # Enviar actualización
            await connection_manager.broadcast_to_session(
                session_id, "streaming_update", update_data
            )

        except Exception as e:
            logger.error(f"❌ Error enviando actualización de streaming: {str(e)}")

    async def _finalize_streaming(self, session_id: str, detection_tracker: StreamingDetectionTracker):
        """Finaliza el streaming y envía resumen final"""
        try:
            session = connection_manager.get_session(session_id)
            if not session:
                return

            # Generar resumen final completo
            final_summary = {
                "session_id": session_id,
                "processing_completed": True,
                "total_processing_time": time.time() - session.start_time if session.start_time else 0,
                "frames_processed": session.processed_frames,
                "frames_with_detections": session.frames_with_detections,
                "detection_summary": detection_tracker.get_streaming_summary(),
                "video_info": session.video_info,
                "processing_params": session.processing_params
            }

            # Actualizar estado de sesión
            connection_manager.update_session_status(session_id, StreamingStatus.COMPLETED)

            # Enviar resumen final
            await connection_manager.send_message(session_id, {
                "type": "streaming_completed",
                "data": final_summary
            })

            logger.success(f"✅ Streaming completado: {session_id}")

        except Exception as e:
            logger.error(f"❌ Error finalizando streaming: {str(e)}")

    async def _handle_streaming_error(self, session_id: str, error_message: str):
        """Maneja errores de streaming"""
        try:
            connection_manager.update_session_status(session_id, StreamingStatus.ERROR)

            await connection_manager.send_message(session_id, {
                "type": "streaming_error",
                "error": error_message,
                "timestamp": time.time()
            })

        except Exception as e:
            logger.error(f"❌ Error manejando error de streaming: {str(e)}")

    def _estimate_processing_time(self, video_info: Dict[str, Any], processing_params: Dict[str, Any]) -> Dict[
        str, Any]:
        """Estima tiempo de procesamiento"""
        try:
            total_frames = video_info["frame_count"]
            frame_skip = processing_params.get("frame_skip", 2)
            frames_to_process = total_frames // frame_skip

            # Estimación base: 0.1 segundos por frame
            base_time_per_frame = 0.1

            # Factores de ajuste
            resolution_factor = 1.0
            width = video_info.get("width", 640)
            if width > 1920:
                resolution_factor = 2.0
            elif width > 1280:
                resolution_factor = 1.5

            device_factor = 0.5 if settings.is_cuda_available else 1.5

            estimated_seconds = frames_to_process * base_time_per_frame * resolution_factor * device_factor

            return {
                "estimated_seconds": round(estimated_seconds, 1),
                "estimated_minutes": round(estimated_seconds / 60, 1),
                "frames_to_process": frames_to_process,
                "factors": {
                    "resolution": resolution_factor,
                    "device": device_factor,
                    "frame_skip": frame_skip
                }
            }

        except Exception as e:
            logger.warning(f"⚠️ Error estimando tiempo: {str(e)}")
            return {"estimated_seconds": 60, "estimated_minutes": 1}

    def _estimate_remaining_time(self, session) -> float:
        """Estima tiempo restante"""
        if not session.start_time or session.processing_speed <= 0:
            return 0.0

        frame_skip = session.processing_params.get("frame_skip", 2)
        total_frames_to_process = session.total_frames // frame_skip
        remaining_frames = total_frames_to_process - session.processed_frames

        return remaining_frames / session.processing_speed

    async def pause_streaming(self, session_id: str) -> bool:
        """Pausa el streaming de una sesión"""
        return await connection_manager.pause_session(session_id)

    async def resume_streaming(self, session_id: str) -> bool:
        """Reanuda el streaming de una sesión"""
        return await connection_manager.resume_session(session_id)

    async def stop_streaming(self, session_id: str) -> bool:
        """Detiene el streaming de una sesión"""
        return await connection_manager.stop_session(session_id)

    def get_streaming_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de streaming"""
        return {
            "active_sessions": len(self.active_sessions),
            "quality_managers": len(self.quality_managers),
            "detection_trackers": len(self.detection_trackers),
            "streaming_config": self.streaming_config,
            "executor_info": {
                "max_workers": self.executor._max_workers,
                "active_threads": len(getattr(self.executor, '_threads', []))
            }
        }

    def cleanup(self):
        """Limpia recursos del servicio"""
        if self.executor:
            self.executor.shutdown(wait=False)

        self.quality_managers.clear()
        self.detection_trackers.clear()
        self.active_sessions.clear()


# Instancia global del servicio de streaming
streaming_service = StreamingVideoProcessor()