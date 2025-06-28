import cv2
import numpy as np
import time
import asyncio
import base64
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import os

from config.settings import settings
from models.model_manager import model_manager
from services.file_service import file_service
from core.utils import PerformanceTimer, get_video_info, format_duration
from core.enhanced_pipeline import EnhancedALPRPipeline  # ✅ USAR PIPELINE MEJORADO


@dataclass
class StreamingFrame:
    """Representa un frame procesado para streaming"""
    frame_num: int
    timestamp: float
    processing_time: float
    detections: List[Dict[str, Any]]
    frame_image_base64: Optional[str] = None
    frame_small_base64: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    original_size: Optional[Tuple[int, int]] = None
    compressed_size: Optional[int] = None
    quality_used: Optional[int] = None
    # ✅ NUEVOS CAMPOS
    roi_used: bool = False
    six_char_filter_applied: bool = False
    six_char_detections_count: int = 0


class AdaptiveQualityManager:
    """Gestor de calidad adaptativa"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_quality = settings.streaming_frame_quality

    def adjust_quality(self, processing_time: float, frame_size: int, connection_speed: float = 1.0) -> int:
        """Ajusta la calidad basándose en métricas de rendimiento"""
        # Si el procesamiento es muy lento, reducir calidad
        if processing_time > 1.0:
            self.current_quality = max(30, self.current_quality - 15)
        elif processing_time > 0.5:
            self.current_quality = max(40, self.current_quality - 10)
        elif processing_time < 0.2:
            self.current_quality = min(85, self.current_quality + 5)

        # Ajustar por tamaño de frame
        if frame_size > 200000:
            self.current_quality = max(25, self.current_quality - 10)
        elif frame_size < 50000:
            self.current_quality = min(90, self.current_quality + 5)

        self.current_quality = max(15, min(95, self.current_quality))
        return self.current_quality


class StreamingDetectionTracker:
    """Tracker de detecciones para streaming con soporte para 6 caracteres"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.unique_plates: Dict[str, Dict[str, Any]] = {}
        self.detection_timeline: List[Dict[str, Any]] = []
        self.frame_detections: Dict[int, List[Dict[str, Any]]] = {}
        self.best_detection_per_plate: Dict[str, Dict[str, Any]] = {}
        # ✅ NUEVOS CONTADORES
        self.six_char_plates: Dict[str, Dict[str, Any]] = {}
        self.total_six_char_detections = 0

    def add_frame_detections(self, frame_num: int, detections: List[Dict[str, Any]], timestamp: float):
        """Agrega detecciones de un frame específico con soporte para 6 caracteres"""
        self.frame_detections[frame_num] = detections

        for detection in detections:
            plate_text = detection["plate_text"]
            confidence = detection["overall_confidence"]
            is_six_char = detection.get("six_char_validated", False)  # ✅ NUEVO

            # Contar detecciones de 6 caracteres
            if is_six_char:
                self.total_six_char_detections += 1

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
                    "is_six_char_valid": False,  # ✅ NUEVO
                    "six_char_detection_count": 0,  # ✅ NUEVO
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

            # ✅ ACTUALIZAR ESTADÍSTICAS DE 6 CARACTERES
            if is_six_char:
                plate_data["is_six_char_valid"] = True
                plate_data["six_char_detection_count"] += 1

                # Agregar a placas de 6 caracteres
                if plate_text not in self.six_char_plates:
                    self.six_char_plates[plate_text] = plate_data

            # Actualizar mejor detección
            if confidence > plate_data["best_confidence"]:
                plate_data["best_confidence"] = confidence
                plate_data["best_frame"] = frame_num
                plate_data["best_timestamp"] = timestamp
                self.best_detection_per_plate[plate_text] = detection

            # Agregar a historial (mantener últimos 20)
            plate_data["frame_history"].append(frame_num)
            plate_data["confidence_trend"].append(confidence)
            if len(plate_data["frame_history"]) > 20:
                plate_data["frame_history"] = plate_data["frame_history"][-20:]
                plate_data["confidence_trend"] = plate_data["confidence_trend"][-20:]

    def get_streaming_summary(self) -> Dict[str, Any]:
        """Genera resumen optimizado para streaming con información de 6 caracteres"""
        sorted_plates = sorted(
            self.unique_plates.values(),
            key=lambda p: (p.get("is_six_char_valid", False), p["best_confidence"]),  # ✅ PRIORIZAR 6 CHARS
            reverse=True
        )

        total_detections = len(self.detection_timeline)
        valid_plates = [p for p in sorted_plates if p["is_valid_format"]]
        six_char_plates = [p for p in sorted_plates if p.get("is_six_char_valid", False)]  # ✅ NUEVO
        frames_with_detections = len(self.frame_detections)

        return {
            "total_detections": total_detections,
            "unique_plates_count": len(self.unique_plates),
            "valid_plates_count": len(valid_plates),
            "six_char_plates_count": len(six_char_plates),  # ✅ NUEVO
            "frames_with_detections": frames_with_detections,
            "best_plates": sorted_plates[:5],
            "best_six_char_plates": six_char_plates[:3],  # ✅ NUEVO
            "latest_detections": self.detection_timeline[-10:],
            "detection_density": frames_with_detections / max(len(self.frame_detections), 1),
            "six_char_detection_rate": self.total_six_char_detections / max(total_detections, 1),  # ✅ NUEVO
            "session_id": self.session_id
        }


class StreamingVideoProcessor:
    """Procesador principal de video con ROI central y filtro de 6 caracteres"""

    def __init__(self):
        self.model_manager = model_manager
        self.file_service = file_service
        # ✅ INICIALIZAR PIPELINE MEJORADO
        self.enhanced_pipeline = EnhancedALPRPipeline(model_manager)
        self.quality_managers: Dict[str, AdaptiveQualityManager] = {}
        self.detection_trackers: Dict[str, StreamingDetectionTracker] = {}
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="streaming_processor")
        self.streaming_config = settings.get_streaming_config()
        logger.info("🎬 StreamingVideoProcessor inicializado con ROI central y filtro de 6 caracteres")

    async def start_video_streaming(
            self,
            session_id: str,
            video_path: str,
            file_info: Dict[str, Any],
            processing_params: Dict[str, Any]
    ) -> bool:
        """Inicia el streaming con ROI central y filtro de 6 caracteres"""
        try:
            logger.info(f"🎬 [STREAMING] Iniciando para sesión: {session_id}")
            logger.info(f"📹 [STREAMING] Video: {video_path}")
            logger.info(f"⚙️ [STREAMING] Params: {processing_params}")

            # Verificaciones básicas
            if not os.path.exists(video_path):
                logger.error(f"❌ [STREAMING] Archivo no existe: {video_path}")
                return False

            if not model_manager.is_loaded:
                logger.error(f"❌ [STREAMING] Modelos no cargados")
                return False

            # Importar get_session del routing
            try:
                from api.routes.streaming import get_session
                session = get_session(session_id)
                if not session:
                    logger.error(f"❌ [STREAMING] Sesión no encontrada: {session_id}")
                    return False
            except ImportError as e:
                logger.error(f"❌ [STREAMING] No se pudo importar get_session: {str(e)}")
                return False

            # Obtener información del video
            logger.info(f"📊 [STREAMING] Obteniendo info del video...")
            video_info = get_video_info(video_path)
            if not video_info:
                logger.error(f"❌ [STREAMING] No se pudo obtener info del video")
                return False

            logger.info(f"📊 [STREAMING] Video info obtenida: {video_info}")

            if video_info.get("frame_count", 0) <= 0:
                logger.error(f"❌ [STREAMING] Video inválido - frame_count: {video_info.get('frame_count')}")
                return False

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
            session.processed_frames = 0
            session.frames_with_detections = 0
            session.total_detection_count = 0
            session.unique_plates = {}
            session.best_detection = None
            session.status = "processing"

            # Enviar información inicial con nuevos campos
            await session.send_message({
                "type": "streaming_started",
                "data": {
                    "message": "Streaming iniciado con ROI central y filtro de 6 caracteres",
                    "video_info": video_info,
                    "file_info": file_info,
                    "processing_params": processing_params,
                    "streaming_config": self.streaming_config,
                    "enhancement_info": {  # ✅ NUEVO
                        "roi_enabled": True,
                        "six_char_filter": True,
                        "roi_percentage": 10.0
                    },
                    "estimated_duration": self._estimate_processing_time(video_info, processing_params)
                }
            })

            logger.info(f"✅ [STREAMING] Información inicial enviada")

            # Iniciar procesamiento en background
            logger.info(f"🎯 [STREAMING] Creando task de procesamiento...")
            asyncio.create_task(self._process_video_stream_enhanced(session_id))

            logger.info(f"✅ [STREAMING] Task de procesamiento creado exitosamente")
            return True

        except Exception as e:
            logger.error(f"❌ [STREAMING] Error iniciando streaming {session_id}: {str(e)}")
            logger.exception("Stack trace completo:")
            await self._handle_streaming_error(session_id, str(e))
            return False

    async def _process_video_stream_enhanced(self, session_id: str):
        """✅ NUEVO: Procesa el video con ROI central y filtro de 6 caracteres"""

        logger.info(f"🎬 [PROCESS] Iniciando procesamiento mejorado para {session_id}")

        try:
            # Importar get_session del routing
            from api.routes.streaming import get_session

            session = get_session(session_id)
            if not session:
                logger.error(f"❌ [PROCESS] Sesión no encontrada: {session_id}")
                return

            video_path = session.video_path
            processing_params = session.processing_params
            quality_manager = self.quality_managers[session_id]
            detection_tracker = self.detection_trackers[session_id]

            logger.info(f"📹 [PROCESS] Abriendo video: {video_path}")

            # Verificar archivo antes de abrir
            if not os.path.exists(video_path):
                raise Exception(f"Archivo de video no existe: {video_path}")

            # Configuración de procesamiento
            frame_skip = processing_params.get("frame_skip", 2)
            max_duration = processing_params.get("max_duration", 600)
            confidence_threshold = processing_params.get("confidence_threshold", 0.3)

            logger.info(
                f"⚙️ [PROCESS] Configuración mejorada: frame_skip={frame_skip}, confidence={confidence_threshold}, ROI=10%, filter_6chars=True")

            # Abrir video con verificación
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"No se pudo abrir el video: {video_path}")

            # Obtener propiedades del video
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                cap.release()
                raise Exception(f"Video inválido - total_frames: {total_frames}")

            logger.info(f"📊 [PROCESS] Video abierto - FPS: {fps}, Frames: {total_frames}")

            # Variables de control
            frame_num = 0
            processed_count = 0
            last_send_time = time.time()
            send_interval = 2.0
            error_count = 0
            max_errors = 10
            six_char_frames = 0  # ✅ CONTADOR NUEVO

            # Enviar primera actualización
            await self._send_initial_update_enhanced(session_id, total_frames)

            # LOOP PRINCIPAL DE PROCESAMIENTO MEJORADO
            logger.info(f"🔄 [PROCESS] Iniciando loop principal mejorado...")

            while True:
                try:
                    # Verificar controles de sesión
                    if session.should_stop:
                        logger.info(f"🛑 [PROCESS] Deteniendo por solicitud: {session_id}")
                        break

                    # Manejar pausa
                    while session.is_paused and not session.should_stop:
                        await asyncio.sleep(0.1)
                        continue

                    # Leer frame con verificación
                    ret, frame = cap.read()
                    if not ret:
                        logger.info(f"📹 [PROCESS] Fin del video alcanzado: {session_id}")
                        break

                    # Verificar límite de duración
                    current_time_in_video = frame_num / fps
                    if current_time_in_video > max_duration:
                        logger.info(f"⏰ [PROCESS] Límite de duración alcanzado: {max_duration}s")
                        break

                    # Procesar solo cada N frames
                    if frame_num % frame_skip == 0:
                        try:
                            # Convertir a RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            logger.debug(f"🔍 [PROCESS] Procesando frame {frame_num} con ROI+6chars")

                            # ✅ PROCESAR FRAME CON MEJORAS
                            streaming_frame = await asyncio.wait_for(
                                self._process_single_frame_enhanced_safe(
                                    session_id, frame_rgb, frame_num, fps, processing_params
                                ),
                                timeout=10.0
                            )

                            processed_count += 1
                            session.processed_frames = processed_count
                            session.current_frame = frame_num

                            # Actualizar velocidad de procesamiento
                            if session.start_time:
                                elapsed = time.time() - session.start_time
                                session.processing_speed = processed_count / max(elapsed, 1)

                            # ✅ PROCESAR DETECCIONES CON INFORMACIÓN DE 6 CARACTERES
                            if streaming_frame.detections:
                                session.frames_with_detections += 1
                                session.total_detection_count += len(streaming_frame.detections)

                                # Contar detecciones de 6 caracteres
                                six_char_count = streaming_frame.six_char_detections_count
                                if six_char_count > 0:
                                    six_char_frames += 1

                                detection_tracker.add_frame_detections(
                                    frame_num, streaming_frame.detections, streaming_frame.timestamp
                                )

                                # Actualizar placas únicas en la sesión
                                session.unique_plates = detection_tracker.unique_plates

                                # Actualizar mejor detección (priorizar 6 caracteres)
                                if detection_tracker.best_detection_per_plate:
                                    # Priorizar detecciones de 6 caracteres
                                    six_char_detections = {
                                        k: v for k, v in detection_tracker.best_detection_per_plate.items()
                                        if v.get("six_char_validated", False)
                                    }

                                    if six_char_detections:
                                        best_overall = max(
                                            six_char_detections.values(),
                                            key=lambda x: x["overall_confidence"]
                                        )
                                    else:
                                        best_overall = max(
                                            detection_tracker.best_detection_per_plate.values(),
                                            key=lambda x: x["overall_confidence"]
                                        )

                                    session.best_detection = best_overall

                                logger.info(
                                    f"🎯 [PROCESS] Frame {frame_num}: {len(streaming_frame.detections)} detecciones "
                                    f"({six_char_count} con 6 chars)")

                            # Enviar datos si es momento adecuado
                            current_time = time.time()
                            should_send = (
                                    current_time - last_send_time >= send_interval or
                                    len(streaming_frame.detections) > 0 or
                                    processed_count % 10 == 0
                            )

                            if should_send:
                                await self._send_streaming_update_enhanced(session_id, streaming_frame,
                                                                           detection_tracker)
                                last_send_time = current_time
                                logger.debug(f"📤 [PROCESS] Actualización mejorada enviada para frame {frame_num}")

                            # Reset error count en caso de éxito
                            error_count = 0

                            # Pequeña pausa para no saturar
                            await asyncio.sleep(0.01)

                        except asyncio.TimeoutError:
                            error_count += 1
                            logger.warning(
                                f"⚠️ [PROCESS] Timeout procesando frame {frame_num} (error {error_count}/{max_errors})")

                            if error_count >= max_errors:
                                raise Exception(f"Demasiados timeouts consecutivos: {error_count}")

                        except Exception as e:
                            error_count += 1
                            logger.warning(
                                f"⚠️ [PROCESS] Error procesando frame {frame_num}: {str(e)} (error {error_count}/{max_errors})")

                            if error_count >= max_errors:
                                raise Exception(f"Demasiados errores consecutivos: {error_count}")

                    frame_num += 1

                    # Log de progreso cada 100 frames
                    if frame_num % 100 == 0:
                        progress = (frame_num / total_frames) * 100
                        six_char_count = len(
                            [p for p in detection_tracker.unique_plates.values() if p.get("is_six_char_valid", False)])
                        logger.info(f"📊 [PROCESS] Progreso {session_id}: {progress:.1f}% - "
                                    f"Placas únicas: {len(detection_tracker.unique_plates)} "
                                    f"(6 chars: {six_char_count})")

                except Exception as frame_error:
                    error_count += 1
                    logger.warning(
                        f"⚠️ [PROCESS] Error en frame {frame_num}: {str(frame_error)} (error {error_count}/{max_errors})")

                    if error_count >= max_errors:
                        raise Exception(f"Demasiados errores en el loop principal: {error_count}")

                    # Continuar con el siguiente frame
                    frame_num += 1
                    continue

            # Finalizar streaming
            cap.release()
            logger.info(f"📹 [PROCESS] Video liberado para {session_id}")

            await self._finalize_streaming_enhanced(session_id, detection_tracker)

        except Exception as e:
            logger.error(f"❌ [PROCESS] Error en streaming mejorado {session_id}: {str(e)}")
            logger.exception("Stack trace completo:")
            await self._handle_streaming_error(session_id, str(e))
        finally:
            # Limpiar recursos
            try:
                if session_id in self.quality_managers:
                    del self.quality_managers[session_id]
                if session_id in self.detection_trackers:
                    del self.detection_trackers[session_id]

                # Limpiar archivo temporal
                if session and hasattr(session, 'video_path') and session.video_path:
                    self.file_service.cleanup_temp_file(session.video_path)

                logger.info(f"🧹 [PROCESS] Recursos limpiados para {session_id}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ [PROCESS] Error en limpieza: {str(cleanup_error)}")

    async def _process_single_frame_enhanced_safe(
            self,
            session_id: str,
            frame: np.ndarray,
            frame_num: int,
            fps: float,
            processing_params: Dict[str, Any]
    ) -> StreamingFrame:
        """✅ NUEVO: Procesa un frame con ROI central y filtro de 6 caracteres"""

        start_time = time.time()
        quality_manager = self.quality_managers[session_id]

        try:
            confidence_threshold = processing_params.get("confidence_threshold", 0.3)
            iou_threshold = processing_params.get("iou_threshold", 0.4)

            # Verificar que el frame es válido
            if frame is None or frame.size == 0:
                raise Exception("Frame inválido o vacío")

            # ✅ PROCESAR CON PIPELINE MEJORADO EN EXECUTOR
            loop = asyncio.get_event_loop()
            pipeline_result = await loop.run_in_executor(
                self.executor,
                self._process_frame_enhanced_sync_safe,
                frame, confidence_threshold, iou_threshold
            )

            # Extraer detecciones con validación mejorada
            detections = []
            six_char_count = 0

            if pipeline_result.get("success") and pipeline_result.get("final_results"):
                for i, plate_result in enumerate(pipeline_result["final_results"]):
                    try:
                        if plate_result.get("plate_text"):
                            is_six_char = plate_result.get("six_char_validated", False)
                            if is_six_char:
                                six_char_count += 1

                            detection = {
                                "detection_id": f"{session_id}_{frame_num}_{i}",
                                "frame_num": frame_num,
                                "timestamp": frame_num / fps,
                                "plate_text": str(plate_result["plate_text"]),
                                "plate_confidence": float(plate_result.get("plate_confidence", 0.0)),
                                "char_confidence": float(
                                    plate_result.get("character_recognition", {}).get("confidence", 0.0)),
                                "overall_confidence": float(plate_result.get("overall_confidence", 0.0)),
                                "plate_bbox": list(plate_result.get("plate_bbox", [0, 0, 0, 0])),
                                "is_valid_plate": bool(plate_result.get("is_valid_plate", False)),
                                "six_char_validated": is_six_char,  # ✅ NUEVO
                                "validation_info": plate_result.get("validation_info", {}),  # ✅ NUEVO
                                "char_count": len(str(plate_result["plate_text"]).replace('-', '').replace(' ', '')),
                                "bbox_area": self._calculate_bbox_area_safe(
                                    plate_result.get("plate_bbox", [0, 0, 0, 0])),
                                "processing_method": "roi_enhanced"  # ✅ MARCADOR
                            }
                            detections.append(detection)
                    except Exception as det_error:
                        logger.warning(f"⚠️ [FRAME] Error procesando detección {i}: {str(det_error)}")

            # Generar imagen para streaming (cada 15 frames o si hay detecciones)
            frame_base64 = None
            frame_small_base64 = None
            compressed_size = 0
            quality_used = quality_manager.current_quality

            try:
                if detections or frame_num % 15 == 0:
                    # Crear frame con detecciones dibujadas (mejorado)
                    annotated_frame = self._draw_detections_enhanced_on_frame_safe(frame, detections)

                    # Codificar frame principal
                    frame_base64, compressed_size, quality_used = self._encode_frame_adaptive_safe(
                        annotated_frame, quality_manager
                    )

                    # Crear thumbnail pequeño
                    frame_small_base64 = self._create_frame_thumbnail_safe(annotated_frame)

                    # Actualizar calidad adaptativa
                    processing_time = time.time() - start_time
                    quality_manager.adjust_quality(processing_time, compressed_size or 0)

            except Exception as encode_error:
                logger.warning(f"⚠️ [FRAME] Error codificando frame {frame_num}: {str(encode_error)}")

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
                success=True,
                # ✅ NUEVOS CAMPOS
                roi_used=True,
                six_char_filter_applied=True,
                six_char_detections_count=six_char_count
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"⚠️ [FRAME] Error procesando frame {frame_num}: {str(e)}")

            return StreamingFrame(
                frame_num=frame_num,
                timestamp=frame_num / fps,
                processing_time=processing_time,
                detections=[],
                success=False,
                error=str(e),
                roi_used=True,
                six_char_filter_applied=True,
                six_char_detections_count=0
            )

    def _process_frame_enhanced_sync_safe(self, frame: np.ndarray, confidence: float, iou: float) -> Dict[str, Any]:
        """✅ NUEVO: Procesamiento síncrono del frame con ROI y filtro 6 chars"""
        try:
            if not self.model_manager.is_loaded:
                return {"success": False, "error": "Modelos no cargados"}

            # ✅ USAR PIPELINE MEJORADO CON ROI Y FILTRO 6 CHARS
            result = self.enhanced_pipeline.process_with_enhancements(
                frame,
                use_roi=True,  # ✅ ACTIVAR ROI CENTRAL
                filter_six_chars=True,  # ✅ ACTIVAR FILTRO 6 CHARS
                return_stats=False,
                conf=confidence,
                iou=iou
            )

            return result

        except Exception as e:
            logger.error(f"❌ [SYNC] Error en pipeline mejorado: {str(e)}")
            return {"success": False, "error": str(e)}

    def _draw_detections_enhanced_on_frame_safe(self, frame: np.ndarray,
                                                detections: List[Dict[str, Any]]) -> np.ndarray:
        """✅ MEJORADO: Dibuja las detecciones con indicadores de 6 caracteres"""
        try:
            annotated_frame = frame.copy()

            for detection in detections:
                try:
                    bbox = detection.get("plate_bbox", [0, 0, 0, 0])
                    plate_text = str(detection.get("plate_text", ""))
                    confidence = float(detection.get("overall_confidence", 0.0))
                    is_valid = bool(detection.get("is_valid_plate", False))
                    is_six_char = bool(detection.get("six_char_validated", False))  # ✅ NUEVO

                    if len(bbox) != 4:
                        continue

                    # Coordenadas
                    x1, y1, x2, y2 = map(int, bbox)

                    # Verificar que las coordenadas están dentro del frame
                    h, w = frame.shape[:2]
                    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                        continue

                    # ✅ COLOR SEGÚN VALIDEZ Y 6 CARACTERES
                    if is_six_char:
                        color = (0, 255, 0)  # Verde brillante para 6 caracteres válidos
                        thickness = 4
                    elif is_valid:
                        color = (255, 255, 0)  # Amarillo para válidos pero no 6 chars
                        thickness = 3
                    elif confidence > 0.5:
                        color = (255, 165, 0)  # Naranja para confianza media
                        thickness = 2
                    else:
                        color = (255, 0, 0)  # Rojo para baja confianza
                        thickness = 2

                    # Dibujar rectángulo
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

                    # ✅ PREPARAR ETIQUETA MEJORADA
                    indicators = []
                    if is_six_char:
                        indicators.append("✅6")
                    elif is_valid:
                        indicators.append("✓")

                    char_count = detection.get("char_count", len(plate_text.replace('-', '').replace(' ', '')))

                    label = f"{plate_text}"
                    if indicators:
                        label += f" {' '.join(indicators)}"
                    label += f" ({char_count}ch)"

                    # Dibujar etiqueta con fondo
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    label_thickness = 2

                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)

                    # Verificar que el texto cabe en el frame
                    if y1 - text_h - 8 >= 0 and x1 + text_w + 8 <= w:
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

                        # ✅ INDICADOR ADICIONAL PARA 6 CARACTERES
                        if is_six_char:
                            # Pequeño círculo verde en la esquina
                            cv2.circle(annotated_frame, (x2 - 8, y1 + 8), 6, (0, 255, 0), -1)

                except Exception as det_error:
                    logger.warning(f"⚠️ Error dibujando detección mejorada: {str(det_error)}")
                    continue

            return annotated_frame

        except Exception as e:
            logger.warning(f"⚠️ Error dibujando detecciones mejoradas: {str(e)}")
            return frame

    # ... [métodos auxiliares sin cambios] ...

    def _calculate_bbox_area_safe(self, bbox: List[float]) -> float:
        """Calcula área del bounding box de forma segura"""
        try:
            if len(bbox) != 4:
                return 0.0
            x1, y1, x2, y2 = bbox
            return max(0.0, abs(x2 - x1) * abs(y2 - y1))
        except Exception:
            return 0.0

    def _encode_frame_adaptive_safe(
            self,
            frame: np.ndarray,
            quality_manager: AdaptiveQualityManager
    ) -> Tuple[str, int, int]:
        """Codifica frame con calidad adaptativa de forma segura"""
        try:
            if frame is None or frame.size == 0:
                return "", 0, 50

            # Redimensionar si es necesario
            config = self.streaming_config["frame_processing"]
            max_size = config.get("max_size", 800)

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
            quality = max(10, min(95, quality_manager.current_quality))

            # Comprimir a JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)

            if success:
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                compressed_size = len(buffer)
                return frame_base64, compressed_size, quality
            else:
                return "", 0, quality

        except Exception as e:
            logger.warning(f"⚠️ Error en codificación adaptativa: {str(e)}")
            return "", 0, 50

    def _create_frame_thumbnail_safe(self, frame: np.ndarray, size: int = 160) -> str:
        """Crea un thumbnail pequeño del frame de forma segura"""
        try:
            if frame is None or frame.size == 0:
                return ""

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

    async def _send_initial_update_enhanced(self, session_id: str, total_frames: int):
        """✅ MEJORADO: Envía actualización inicial con información de mejoras"""
        try:
            from api.routes.streaming import get_session

            session = get_session(session_id)
            if session:
                await session.send_message({
                    "type": "streaming_update",
                    "data": {
                        "progress": {
                            "processed_frames": 0,
                            "total_frames": total_frames,
                            "progress_percent": 0.0,
                            "processing_speed": 0.0
                        },
                        "frame_info": {
                            "frame_number": 0,
                            "timestamp": 0.0,
                            "processing_time": 0.0,
                            "success": True,
                            "roi_used": True,  # ✅ NUEVO
                            "six_char_filter_applied": True  # ✅ NUEVO
                        },
                        "detection_summary": {
                            "total_detections": 0,
                            "unique_plates_count": 0,
                            "valid_plates_count": 0,
                            "six_char_plates_count": 0,  # ✅ NUEVO
                            "frames_with_detections": 0,
                            "best_plates": [],
                            "best_six_char_plates": [],  # ✅ NUEVO
                            "latest_detections": [],
                            "six_char_detection_rate": 0.0  # ✅ NUEVO
                        },
                        "enhancement_info": {  # ✅ INFORMACIÓN DE MEJORAS
                            "roi_enabled": True,
                            "roi_percentage": 10.0,
                            "six_char_filter": True,
                            "processing_method": "roi_enhanced"
                        }
                    }
                })
                logger.debug(f"📤 [INITIAL] Actualización inicial mejorada enviada para {session_id}")
        except Exception as e:
            logger.warning(f"⚠️ [INITIAL] Error enviando actualización inicial mejorada: {str(e)}")

    async def _send_streaming_update_enhanced(
            self,
            session_id: str,
            streaming_frame: StreamingFrame,
            detection_tracker: StreamingDetectionTracker
    ):
        """✅ MEJORADO: Envía actualización de streaming con información de 6 caracteres"""
        try:
            from api.routes.streaming import get_session

            session = get_session(session_id)
            if not session:
                logger.warning(f"⚠️ [UPDATE] Sesión no encontrada para envío: {session_id}")
                return

            # Calcular progreso de forma segura
            progress_percent = 0
            if session.total_frames > 0:
                frames_to_process = session.total_frames // session.processing_params.get("frame_skip", 2)
                progress_percent = (session.processed_frames / max(frames_to_process, 1)) * 100

            # Obtener resumen de detecciones mejorado
            detection_summary = detection_tracker.get_streaming_summary()

            # Preparar datos de actualización mejorados
            update_data = {
                "frame_info": {
                    "frame_number": streaming_frame.frame_num,
                    "timestamp": streaming_frame.timestamp,
                    "processing_time": streaming_frame.processing_time,
                    "success": streaming_frame.success,
                    "roi_used": streaming_frame.roi_used,  # ✅ NUEVO
                    "six_char_filter_applied": streaming_frame.six_char_filter_applied,  # ✅ NUEVO
                    "six_char_detections_in_frame": streaming_frame.six_char_detections_count  # ✅ NUEVO
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
                },
                "enhancement_stats": {  # ✅ ESTADÍSTICAS DE MEJORAS
                    "roi_processing": True,
                    "six_char_filter_active": True,
                    "total_six_char_detections": detection_tracker.total_six_char_detections,
                    "six_char_plates_found": len(detection_tracker.six_char_plates),
                    "six_char_detection_rate": detection_summary.get("six_char_detection_rate", 0.0)
                }
            }

            # Incluir frame si existe
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
                    "recommended_frame_skip": 2,
                    "adaptive_enabled": self.streaming_config["frame_processing"]["adaptive_quality"]
                }

            # Enviar actualización
            success = await session.send_message({
                "type": "streaming_update",
                "data": update_data,
                "timestamp": time.time()
            })

            if not success:
                logger.warning(f"⚠️ [UPDATE] No se pudo enviar actualización a {session_id}")
            else:
                logger.debug(f"✅ [UPDATE] Actualización mejorada enviada exitosamente a {session_id}")

        except Exception as e:
            logger.error(f"❌ [UPDATE] Error enviando actualización de streaming mejorada: {str(e)}")

    async def _finalize_streaming_enhanced(self, session_id: str, detection_tracker: StreamingDetectionTracker):
        """✅ MEJORADO: Finaliza el streaming con resumen completo de mejoras"""
        try:
            from api.routes.streaming import get_session

            session = get_session(session_id)
            if not session:
                logger.warning(f"⚠️ [FINALIZE] Sesión no encontrada para finalización: {session_id}")
                return

            # Generar resumen final completo con estadísticas de mejoras
            detection_summary = detection_tracker.get_streaming_summary()
            six_char_plates = [p for p in detection_tracker.unique_plates.values() if p.get("is_six_char_valid", False)]

            final_summary = {
                "session_id": session_id,
                "processing_completed": True,
                "total_processing_time": time.time() - session.start_time if session.start_time else 0,
                "frames_processed": session.processed_frames,
                "frames_with_detections": session.frames_with_detections,
                "detection_summary": detection_summary,
                "video_info": session.video_info,
                "processing_params": session.processing_params,
                # ✅ RESUMEN DE MEJORAS
                "enhancement_summary": {
                    "roi_processing": True,
                    "roi_percentage": 10.0,
                    "six_char_filter": True,
                    "total_six_char_detections": detection_tracker.total_six_char_detections,
                    "six_char_plates_found": len(six_char_plates),
                    "six_char_success_rate": (len(six_char_plates) / max(len(detection_tracker.unique_plates),
                                                                         1)) * 100,
                    "processing_method": "roi_enhanced"
                },
                # ✅ MEJORES PLACAS DE 6 CARACTERES
                "best_six_char_plates": sorted(
                    six_char_plates,
                    key=lambda p: p["best_confidence"],
                    reverse=True
                )[:3]
            }

            # Actualizar estado de sesión
            session.status = "completed"

            # Enviar resumen final
            await session.send_message({
                "type": "streaming_completed",
                "data": final_summary
            })

            six_char_count = len(six_char_plates)
            total_plates = len(detection_tracker.unique_plates)

            logger.success(f"✅ [FINALIZE] Streaming mejorado completado: {session_id} - "
                           f"Placas: {total_plates} total, {six_char_count} con 6 caracteres válidos "
                           f"({(six_char_count / max(total_plates, 1) * 100):.1f}% éxito), "
                           f"Frames: {session.processed_frames}")

        except Exception as e:
            logger.error(f"❌ [FINALIZE] Error finalizando streaming mejorado: {str(e)}")

    # ... [resto de métodos auxiliares sin cambios] ...

    async def _handle_streaming_error(self, session_id: str, error_message: str):
        """Maneja errores de streaming"""
        try:
            from api.routes.streaming import get_session

            session = get_session(session_id)
            if session:
                session.status = "error"
                await session.send_message({
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
        try:
            if not session.start_time or session.processing_speed <= 0:
                return 0.0

            frame_skip = session.processing_params.get("frame_skip", 2)
            total_frames_to_process = session.total_frames // frame_skip
            remaining_frames = total_frames_to_process - session.processed_frames

            return max(0.0, remaining_frames / session.processing_speed)
        except Exception:
            return 0.0

    # ... [métodos de control sin cambios] ...

    async def pause_streaming(self, session_id: str) -> bool:
        """Pausa el streaming de una sesión"""
        try:
            from api.routes.streaming import get_session
            session = get_session(session_id)
            if session:
                session.is_paused = True
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error pausando streaming: {str(e)}")
            return False

    async def resume_streaming(self, session_id: str) -> bool:
        """Reanuda el streaming de una sesión"""
        try:
            from api.routes.streaming import get_session
            session = get_session(session_id)
            if session:
                session.is_paused = False
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error reanudando streaming: {str(e)}")
            return False

    async def stop_streaming(self, session_id: str) -> bool:
        """Detiene el streaming de una sesión"""
        try:
            from api.routes.streaming import get_session
            session = get_session(session_id)
            if session:
                session.should_stop = True
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error deteniendo streaming: {str(e)}")
            return False

    def cleanup(self):
        """Limpia recursos del servicio"""
        try:
            if self.executor:
                self.executor.shutdown(wait=False)

            self.quality_managers.clear()
            self.detection_trackers.clear()

            logger.info("🧹 StreamingVideoProcessor mejorado limpiado")
        except Exception as e:
            logger.warning(f"⚠️ Error en cleanup: {str(e)}")


# Instancia global del servicio de streaming mejorado
streaming_service = StreamingVideoProcessor()