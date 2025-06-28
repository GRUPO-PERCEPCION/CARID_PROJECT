import cv2
import numpy as np
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings
from models.model_manager import model_manager
from services.file_service import file_service
from core.utils import PerformanceTimer
from core.enhanced_pipeline import EnhancedALPRPipeline


@dataclass
class PlateTracker:
    """Clase básica para trackear placas detectadas y evitar duplicados"""
    plate_text: str
    best_confidence: float
    best_frame: int
    detection_count: int
    first_seen: int
    last_seen: int
    bbox_history: List[List[float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    is_six_char_valid: bool = False  # ✅ NUEVO: Validación de 6 caracteres

    def update(self, confidence: float, frame_num: int, bbox: List[float], is_six_char: bool = False):
        """Actualiza el tracker con nueva detección"""
        self.detection_count += 1
        self.last_seen = frame_num
        self.bbox_history.append(bbox)
        self.confidences.append(confidence)

        # Actualizar validación de 6 caracteres
        if is_six_char:
            self.is_six_char_valid = True

        # Actualizar mejor detección
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_frame = frame_num

    def is_similar_position(self, new_bbox: List[float], threshold: float = 0.3) -> bool:
        """Verifica si una nueva detección está en posición similar"""
        if not self.bbox_history:
            return False

        # Calcular IoU con el último bbox
        last_bbox = self.bbox_history[-1]
        iou = self._calculate_iou(last_bbox, new_bbox)
        return iou > threshold

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calcula Intersection over Union entre dos bboxes"""
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


class VideoService:
    """Servicio para procesamiento de videos con ROI central y filtro de 6 caracteres"""

    def __init__(self):
        self.model_manager = model_manager
        self.file_service = file_service

        # ✅ INICIALIZAR PIPELINE MEJORADO
        self.enhanced_pipeline = EnhancedALPRPipeline(model_manager)

        # Configuración de procesamiento
        self.frame_skip = getattr(settings, 'video_frame_skip', 3)
        self.similarity_threshold = getattr(settings, 'video_similarity_threshold', 0.7)
        self.min_detection_frames = getattr(settings, 'video_min_detection_frames', 2)
        self.max_tracking_distance = getattr(settings, 'video_max_tracking_distance', 5)

        # Thread pool para procesamiento paralelo
        self.executor = ThreadPoolExecutor(max_workers=2)

        logger.info("🎬 VideoService inicializado con ROI central y filtro de 6 caracteres")

    async def process_video(
            self,
            video_path: str,
            file_info: Dict[str, Any],
            request_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa un video completo con ROI central y filtro de 6 caracteres
        """
        start_time = time.time()
        result_id = self.file_service.create_result_id()

        try:
            logger.info(f"🎬 Iniciando procesamiento de video: {file_info['filename']}")

            # Verificar que los modelos estén cargados
            if not self.model_manager.is_loaded:
                raise Exception("Los modelos no están cargados")

            # Obtener información del video
            video_info = self._get_video_info(video_path)
            if not video_info:
                raise Exception("No se pudo obtener información del video")

            logger.info(f"📹 Video: {video_info['duration']:.1f}s, "
                        f"{video_info['total_frames']} frames, "
                        f"{video_info['fps']:.1f} FPS")

            # Verificar duración máxima
            max_duration = request_params.get('max_duration', getattr(settings, 'max_video_duration', 300))
            if video_info['duration'] > max_duration:
                raise Exception(f"Video muy largo. Máximo: {max_duration}s, "
                                f"recibido: {video_info['duration']:.1f}s")

            # Configurar parámetros de procesamiento
            self.frame_skip = request_params.get('frame_skip', self.frame_skip)

            # ✅ PROCESAR CON ROI Y FILTRO DE 6 CARACTERES
            with PerformanceTimer("Procesamiento de video con ROI + 6 chars"):
                tracking_result = await self._process_video_with_roi_and_filter(
                    video_path, video_info, request_params
                )
                unique_plates = self._extract_unique_plates_enhanced(tracking_result['trackers'])

            # Guardar resultados si se solicita
            result_urls = {}
            if request_params.get('save_results', True):
                try:
                    # Copiar video original
                    original_path = await self.file_service.copy_to_results(
                        video_path, result_id, "original"
                    )
                    result_urls["original"] = self.file_service.get_file_url(original_path)

                    # Guardar frames con mejores detecciones
                    if request_params.get('save_best_frames', True):
                        frames_urls = await self._save_best_frames(
                            video_path, unique_plates, result_id
                        )
                        result_urls.update(frames_urls)

                except Exception as e:
                    logger.warning(f"⚠️ Error guardando resultados: {str(e)}")

            # Tiempo de procesamiento
            processing_time = time.time() - start_time

            # Crear resultado final
            result = {
                "success": len(unique_plates) > 0,
                "message": self._generate_video_result_message(unique_plates, tracking_result),
                "file_info": file_info,
                "video_info": video_info,
                "processing_summary": {
                    "frames_processed": tracking_result['frames_processed'],
                    "frames_with_detections": tracking_result['frames_with_detections'],
                    "total_detections": tracking_result['total_detections'],
                    "unique_plates_found": len(unique_plates),
                    "valid_plates": len([p for p in unique_plates if self._get_is_valid(p)]),
                    "six_char_plates": len([p for p in unique_plates if p.get('is_six_char_valid', False)])  # ✅ NUEVO
                },
                "unique_plates": unique_plates,
                "best_plate": unique_plates[0] if unique_plates else None,
                "processing_time": round(processing_time, 3),
                "timestamp": time.time(),
                "result_urls": result_urls if result_urls else None,
                "tracking_type": "enhanced_roi_6chars",  # ✅ NUEVO TIPO
                "enhancement_info": {  # ✅ INFORMACIÓN ADICIONAL
                    "roi_enabled": True,
                    "six_char_filter": True,
                    "roi_percentage": 10.0
                }
            }

            # Limpiar archivo temporal
            self.file_service.cleanup_temp_file(video_path)

            logger.success(f"✅ Video procesado en {processing_time:.3f}s. "
                           f"Placas únicas encontradas: {len(unique_plates)} "
                           f"(6 chars válidas: {len([p for p in unique_plates if p.get('is_six_char_valid', False)])})")

            return result

        except Exception as e:
            logger.error(f"❌ Error procesando video: {str(e)}")

            # Limpiar archivo temporal
            self.file_service.cleanup_temp_file(video_path)

            processing_time = time.time() - start_time

            return {
                "success": False,
                "message": f"Error procesando video: {str(e)}",
                "file_info": file_info,
                "video_info": self._get_video_info(video_path) or {},
                "processing_summary": {
                    "frames_processed": 0,
                    "frames_with_detections": 0,
                    "total_detections": 0,
                    "unique_plates_found": 0,
                    "valid_plates": 0,
                    "six_char_plates": 0
                },
                "unique_plates": [],
                "best_plate": None,
                "processing_time": round(processing_time, 3),
                "timestamp": time.time(),
                "result_urls": None,
                "tracking_type": "error"
            }

    async def _process_video_with_roi_and_filter(
            self,
            video_path: str,
            video_info: Dict[str, Any],
            request_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """✅ NUEVO: Procesa video con ROI central y filtro de 6 caracteres"""

        plate_trackers: Dict[str, PlateTracker] = {}
        frame_results = []

        frames_processed = 0
        frames_with_detections = 0
        total_detections = 0
        six_char_detections = 0  # ✅ CONTADOR NUEVO

        cap = cv2.VideoCapture(video_path)
        frame_num = 0

        # Configurar parámetros para el modelo
        confidence_threshold = request_params.get('confidence_threshold', 0.4)
        iou_threshold = request_params.get('iou_threshold', 0.4)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Procesar solo cada N frames
                if frame_num % self.frame_skip == 0:
                    try:
                        # Convertir frame a RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # ✅ PROCESAR CON PIPELINE MEJORADO (ROI + FILTRO 6 CHARS)
                        frame_result = self._process_single_frame_with_enhancements(
                            frame_rgb, frame_num, confidence_threshold, iou_threshold
                        )

                        frames_processed += 1

                        if frame_result['detections']:
                            frames_with_detections += 1
                            total_detections += len(frame_result['detections'])

                            # Contar detecciones de 6 caracteres
                            six_char_count = len(
                                [d for d in frame_result['detections'] if d.get('six_char_validated', False)])
                            six_char_detections += six_char_count

                            # Actualizar trackers con nuevas detecciones
                            self._update_trackers_enhanced(
                                plate_trackers,
                                frame_result['detections'],
                                frame_num
                            )

                        frame_results.append(frame_result)

                        # Log de progreso cada 30 frames procesados
                        if frames_processed % 30 == 0:
                            progress = (frame_num / video_info['total_frames']) * 100
                            six_char_plates = len([t for t in plate_trackers.values() if t.is_six_char_valid])
                            logger.info(f"📊 Progreso: {progress:.1f}% - "
                                        f"Placas únicas: {len(plate_trackers)} "
                                        f"(6 chars: {six_char_plates})")

                    except Exception as e:
                        logger.warning(f"⚠️ Error procesando frame {frame_num}: {str(e)}")

                frame_num += 1

                # Verificar límite de tiempo
                if frame_num > video_info['total_frames']:
                    break

        finally:
            cap.release()

        return {
            "trackers": plate_trackers,
            "frame_results": frame_results,
            "frames_processed": frames_processed,
            "frames_with_detections": frames_with_detections,
            "total_detections": total_detections,
            "six_char_detections": six_char_detections  # ✅ NUEVO
        }

    def _process_single_frame_with_enhancements(
            self,
            frame: np.ndarray,
            frame_num: int,
            confidence_threshold: float,
            iou_threshold: float
    ) -> Dict[str, Any]:
        """
        ✅ NUEVO: Procesa un frame con ROI central y filtro de 6 caracteres
        """
        try:
            # ✅ PROCESAR CON PIPELINE MEJORADO
            result = self.enhanced_pipeline.process_with_enhancements(
                frame,
                use_roi=True,  # ✅ ACTIVAR ROI CENTRAL
                filter_six_chars=True,  # ✅ ACTIVAR FILTRO 6 CHARS
                return_stats=False,
                conf=confidence_threshold,
                iou=iou_threshold
            )

            # Extraer detecciones válidas
            detections = []
            if result.get("success") and result.get("final_results"):
                for plate_result in result["final_results"]:
                    if plate_result["plate_text"]:
                        detection = {
                            "plate_text": plate_result["plate_text"],
                            "confidence": plate_result["overall_confidence"],
                            "plate_bbox": plate_result["plate_bbox"],
                            "plate_confidence": plate_result["plate_confidence"],
                            "char_confidence": plate_result.get("character_recognition", {}).get("confidence", 0.0),
                            "is_valid_plate": plate_result["is_valid_plate"],
                            "six_char_validated": plate_result.get("six_char_validated", False),  # ✅ NUEVO
                            "validation_info": plate_result.get("validation_info", {}),  # ✅ NUEVO
                            "frame_num": frame_num,
                            "processing_method": "roi"  # ✅ MARCADOR
                        }
                        detections.append(detection)

            return {
                "frame_num": frame_num,
                "detections": detections,
                "processing_success": result.get("success", False),
                "roi_used": result.get("use_roi", False),  # ✅ NUEVO
                "filter_applied": result.get("filter_six_chars", False)  # ✅ NUEVO
            }

        except Exception as e:
            logger.warning(f"⚠️ Error en frame {frame_num}: {str(e)}")
            return {
                "frame_num": frame_num,
                "detections": [],
                "processing_success": False,
                "error": str(e),
                "roi_used": True,
                "filter_applied": True
            }

    def _update_trackers_enhanced(
            self,
            trackers: Dict[str, PlateTracker],
            detections: List[Dict[str, Any]],
            frame_num: int
    ):
        """✅ ACTUALIZADO: Actualiza trackers con información de 6 caracteres"""
        for detection in detections:
            plate_text = detection["plate_text"]
            confidence = detection["confidence"]
            bbox = detection["plate_bbox"]
            is_six_char = detection.get("six_char_validated", False)  # ✅ NUEVO

            # Buscar tracker existente para esta placa
            existing_tracker = None

            # Buscar por texto exacto primero
            if plate_text in trackers:
                existing_tracker = trackers[plate_text]
            else:
                # Buscar por similitud de texto y posición
                for tracked_text, tracker in trackers.items():
                    if (self._are_plates_similar(plate_text, tracked_text) and
                            tracker.is_similar_position(bbox)):
                        existing_tracker = tracker
                        break

            if existing_tracker:
                # Actualizar tracker existente
                existing_tracker.update(confidence, frame_num, bbox, is_six_char)
                logger.debug(f"✅ Placa actualizada: {plate_text} (Frame: {frame_num}, 6chars: {is_six_char})")
            else:
                # Crear nuevo tracker
                trackers[plate_text] = PlateTracker(
                    plate_text=plate_text,
                    best_confidence=confidence,
                    best_frame=frame_num,
                    detection_count=1,
                    first_seen=frame_num,
                    last_seen=frame_num,
                    is_six_char_valid=is_six_char  # ✅ NUEVO
                )
                trackers[plate_text].update(confidence, frame_num, bbox, is_six_char)
                logger.debug(f"🆕 Nueva placa detectada: {plate_text} (Frame: {frame_num}, 6chars: {is_six_char})")

    def _extract_unique_plates_enhanced(self, trackers: Dict[str, PlateTracker]) -> List[Dict[str, Any]]:
        """✅ ACTUALIZADO: Extrae placas únicas con información de 6 caracteres"""
        unique_plates = []

        for plate_text, tracker in trackers.items():
            # Filtrar trackers con pocas detecciones
            if tracker.detection_count >= self.min_detection_frames:
                avg_confidence = sum(tracker.confidences) / len(tracker.confidences) if tracker.confidences else 0.0
                stability_score = 1.0 - (np.std(tracker.confidences) / np.mean(tracker.confidences)) if len(
                    tracker.confidences) > 1 and np.mean(tracker.confidences) > 0 else 0.5

                plate_result = {
                    "plate_text": tracker.plate_text,
                    "best_confidence": tracker.best_confidence,
                    "detection_count": tracker.detection_count,
                    "first_seen_frame": tracker.first_seen,
                    "last_seen_frame": tracker.last_seen,
                    "best_frame": tracker.best_frame,
                    "is_valid_format": self._validate_plate_format(tracker.plate_text),
                    "is_six_char_valid": tracker.is_six_char_valid,  # ✅ NUEVO
                    "avg_confidence": round(avg_confidence, 3),
                    "stability_score": round(stability_score, 3),
                    "duration_frames": tracker.last_seen - tracker.first_seen + 1,
                    "char_count": len(tracker.plate_text.replace('-', '').replace(' ', '')),  # ✅ NUEVO
                    "processing_method": "roi_enhanced"  # ✅ MARCADOR
                }
                unique_plates.append(plate_result)

                status_indicator = "✅" if tracker.is_six_char_valid else "❌"
                logger.info(f"📋 Placa confirmada: '{tracker.plate_text}' {status_indicator} "
                            f"(Detecciones: {tracker.detection_count}, "
                            f"Confianza: {tracker.best_confidence:.3f}, "
                            f"6chars: {tracker.is_six_char_valid})")

        # Ordenar por: 1) Validez de 6 chars, 2) Confianza
        unique_plates.sort(key=lambda x: (x["is_six_char_valid"], x["best_confidence"]), reverse=True)
        return unique_plates

    # ... [resto de métodos sin cambios] ...

    def _get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Obtiene información detallada del video"""
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
                "total_frames": frame_count,
                "width": width,
                "height": height,
                "duration": duration,
                "frames_to_process": frame_count // self.frame_skip,
                "file_size_mb": self.file_service.get_file_size_mb(video_path)
            }

        except Exception as e:
            logger.error(f"❌ Error obteniendo info del video: {str(e)}")
            return None

    def _are_plates_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Verifica si dos textos de placa son similares"""
        if not text1 or not text2:
            return False

        # Normalizar textos
        text1 = text1.upper().replace('-', '').replace(' ', '')
        text2 = text2.upper().replace('-', '').replace(' ', '')

        if text1 == text2:
            return True

        # Calcular similitud usando caracteres
        if len(text1) != len(text2):
            return False

        differences = sum(c1 != c2 for c1, c2 in zip(text1, text2))
        similarity = 1 - (differences / len(text1))

        return similarity >= threshold

    def _validate_plate_format(self, plate_text: str) -> bool:
        """Valida formato de placa peruana"""
        import re
        patterns = [
            r'^[A-Z]{3}-\d{3}$',  # ABC-123
            r'^[A-Z]{2}-\d{4}$',  # AB-1234
            r'^[A-Z]\d{2}-\d{3}$',  # A12-345
            r'^[A-Z]{3}\d{3}$',  # ABC123 (sin guión)
        ]
        return any(re.match(pattern, plate_text) for pattern in patterns)

    def _get_is_valid(self, plate: Dict[str, Any]) -> bool:
        """Obtiene validez de placa según el tipo de resultado"""
        return plate.get("is_valid_format", False)

    def _generate_video_result_message(
            self,
            unique_plates: List[Dict[str, Any]],
            tracking_result: Dict[str, Any]
    ) -> str:
        """Genera mensaje descriptivo del resultado"""
        if not unique_plates:
            return f"No se detectaron placas válidas en el video. " \
                   f"Frames procesados: {tracking_result['frames_processed']}, " \
                   f"detecciones totales: {tracking_result['total_detections']}"

        valid_plates = [p for p in unique_plates if p['is_valid_format']]
        six_char_plates = [p for p in unique_plates if p.get('is_six_char_valid', False)]
        best_plate = unique_plates[0]

        message = f"Se detectaron {len(unique_plates)} placa(s) única(s). "

        if six_char_plates:
            message += f"{len(six_char_plates)} con 6 caracteres válidos. "

        message += f"Mejor: '{best_plate['plate_text']}' " \
                   f"(Confianza: {best_plate['best_confidence']:.3f}, " \
                   f"Detecciones: {best_plate['detection_count']}"

        if best_plate.get('is_six_char_valid'):
            message += ", ✅ 6 chars"

        message += ")"

        return message

    async def _save_best_frames(
            self,
            video_path: str,
            unique_plates: List[Dict[str, Any]],
            result_id: str
    ) -> Dict[str, str]:
        """Guarda los frames con mejores detecciones"""
        frame_urls = {}

        try:
            cap = cv2.VideoCapture(video_path)

            for i, plate in enumerate(unique_plates[:5]):  # Máximo 5 mejores placas
                try:
                    # best_frame es solo el número
                    best_frame_num = plate.get('best_frame', 0)

                    # Ir al frame con mejor detección
                    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_num)
                    ret, frame = cap.read()

                    if ret:
                        # Convertir a RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Guardar frame
                        frame_path = await self.file_service.save_result_image(
                            frame_rgb,
                            result_id,
                            f"best_frame_{i + 1}_{plate['plate_text'].replace('-', '_')}"
                        )

                        frame_urls[f"best_frame_{i + 1}"] = self.file_service.get_file_url(frame_path)

                        logger.info(f"💾 Frame guardado: {plate['plate_text']} (Frame: {best_frame_num})")

                except Exception as e:
                    logger.warning(f"⚠️ Error guardando frame para placa {plate['plate_text']}: {str(e)}")

            cap.release()

        except Exception as e:
            logger.warning(f"⚠️ Error guardando frames: {str(e)}")

        return frame_urls


# Instancia global del servicio
video_service = VideoService()