"""
Servicio para procesamiento de videos con reconocimiento inteligente de placas
"""

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


@dataclass
class PlateTracker:
    """Clase para trackear placas detectadas y evitar duplicados"""
    plate_text: str
    best_confidence: float
    best_frame: int
    detection_count: int
    first_seen: int
    last_seen: int
    bbox_history: List[List[float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    def update(self, confidence: float, frame_num: int, bbox: List[float]):
        """Actualiza el tracker con nueva detección"""
        self.detection_count += 1
        self.last_seen = frame_num
        self.bbox_history.append(bbox)
        self.confidences.append(confidence)

        # Actualizar mejor detección
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_frame = frame_num

    def get_average_bbox(self) -> List[float]:
        """Calcula el bbox promedio de todas las detecciones"""
        if not self.bbox_history:
            return [0, 0, 0, 0]

        avg_bbox = [
            sum(bbox[i] for bbox in self.bbox_history) / len(self.bbox_history)
            for i in range(4)
        ]
        return avg_bbox

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
    """Servicio para procesamiento de videos con reconocimiento inteligente"""

    def __init__(self):
        self.model_manager = model_manager
        self.file_service = file_service

        # Configuración de procesamiento
        self.frame_skip = 3  # Procesar cada N frames para optimizar
        self.similarity_threshold = 0.7  # Umbral para considerar placas similares
        self.min_detection_frames = 2  # Mínimo frames para confirmar una placa
        self.max_tracking_distance = 5  # Máximo frames sin ver la placa

        # Thread pool para procesamiento paralelo
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def process_video(
            self,
            video_path: str,
            file_info: Dict[str, Any],
            request_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa un video completo con reconocimiento inteligente de placas

        Args:
            video_path: Ruta del video
            file_info: Información del archivo
            request_params: Parámetros de la solicitud

        Returns:
            Resultado completo del procesamiento
        """
        start_time = time.time()
        result_id = self.file_service.create_result_id()

        try:
            logger.info(f"🎬 Iniciando procesamiento de video: {file_info['filename']}")

            # Verificar modelos
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
            max_duration = request_params.get('max_duration', settings.max_video_duration)
            if video_info['duration'] > max_duration:
                raise Exception(f"Video muy largo. Máximo: {max_duration}s, "
                                f"recibido: {video_info['duration']:.1f}s")

            # Configurar parámetros de procesamiento
            self.frame_skip = request_params.get('frame_skip', self.frame_skip)

            # Procesar video frame por frame
            with PerformanceTimer("Procesamiento de video"):
                tracking_result = await self._process_video_with_tracking(
                    video_path, video_info, request_params
                )

            # Filtrar y validar placas únicas
            unique_plates = self._extract_unique_plates(tracking_result['trackers'])

            # Guardar resultados si se solicita
            result_urls = {}
            if request_params.get('save_results', True):
                try:
                    # Copiar video original
                    original_path = await self.file_service.copy_to_results(
                        video_path, result_id, "original"
                    )
                    result_urls["original"] = self.file_service.get_file_url(original_path)

                    # Crear video con anotaciones si se solicita
                    if request_params.get('create_annotated_video', False):
                        annotated_path = await self._create_annotated_video(
                            video_path, tracking_result['frame_results'], result_id
                        )
                        if annotated_path:
                            result_urls["annotated"] = self.file_service.get_file_url(annotated_path)

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
                    "valid_plates": len([p for p in unique_plates if p['is_valid_format']])
                },
                "unique_plates": unique_plates,
                "best_plate": unique_plates[0] if unique_plates else None,
                "processing_time": round(processing_time, 3),
                "timestamp": time.time(),
                "result_urls": result_urls if result_urls else None
            }

            # Limpiar archivo temporal
            self.file_service.cleanup_temp_file(video_path)

            logger.success(f"✅ Video procesado en {processing_time:.3f}s. "
                           f"Placas únicas encontradas: {len(unique_plates)}")

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
                    "valid_plates": 0
                },
                "unique_plates": [],
                "best_plate": None,
                "processing_time": round(processing_time, 3),
                "timestamp": time.time(),
                "result_urls": None
            }

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
                "frames_to_process": frame_count // self.frame_skip
            }

        except Exception as e:
            logger.error(f"❌ Error obteniendo info del video: {str(e)}")
            return None

    async def _process_video_with_tracking(
            self,
            video_path: str,
            video_info: Dict[str, Any],
            request_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Procesa video frame por frame con tracking de placas"""

        # Diccionario para trackear placas por texto
        plate_trackers: Dict[str, PlateTracker] = {}
        frame_results = []

        frames_processed = 0
        frames_with_detections = 0
        total_detections = 0

        # Configurar parámetros del modelo
        model_kwargs = {
            'conf': request_params.get('confidence_threshold', 0.4),  # Más permisivo para videos
            'iou': request_params.get('iou_threshold', 0.4),
            'verbose': False
        }

        cap = cv2.VideoCapture(video_path)
        frame_num = 0

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

                        # Procesar frame con el pipeline completo
                        frame_result = await self._process_single_frame(
                            frame_rgb, frame_num, model_kwargs
                        )

                        frames_processed += 1

                        if frame_result['detections']:
                            frames_with_detections += 1
                            total_detections += len(frame_result['detections'])

                            # Actualizar trackers con nuevas detecciones
                            self._update_trackers(
                                plate_trackers,
                                frame_result['detections'],
                                frame_num
                            )

                        frame_results.append(frame_result)

                        # Log de progreso cada 30 frames procesados
                        if frames_processed % 30 == 0:
                            progress = (frame_num / video_info['total_frames']) * 100
                            logger.info(f"📊 Progreso: {progress:.1f}% - "
                                        f"Placas únicas: {len(plate_trackers)}")

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
            "total_detections": total_detections
        }

    async def _process_single_frame(
            self,
            frame: np.ndarray,
            frame_num: int,
            model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Procesa un frame individual"""
        try:
            # Ejecutar pipeline en thread separado para no bloquear
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.model_manager.process_full_pipeline,
                frame,
                **model_kwargs
            )

            # Extraer detecciones válidas
            detections = []
            if result.get("success") and result.get("final_results"):
                for plate_result in result["final_results"]:
                    if plate_result["plate_text"]:  # Solo placas con texto
                        detection = {
                            "plate_text": plate_result["plate_text"],
                            "confidence": plate_result["overall_confidence"],
                            "bbox": plate_result["plate_bbox"],
                            "is_valid_format": plate_result["is_valid_plate"],
                            "frame_num": frame_num
                        }
                        detections.append(detection)

            return {
                "frame_num": frame_num,
                "detections": detections,
                "processing_success": result.get("success", False)
            }

        except Exception as e:
            logger.warning(f"⚠️ Error en frame {frame_num}: {str(e)}")
            return {
                "frame_num": frame_num,
                "detections": [],
                "processing_success": False,
                "error": str(e)
            }

    def _update_trackers(
            self,
            trackers: Dict[str, PlateTracker],
            detections: List[Dict[str, Any]],
            frame_num: int
    ):
        """Actualiza los trackers con nuevas detecciones"""

        for detection in detections:
            plate_text = detection["plate_text"]
            confidence = detection["confidence"]
            bbox = detection["bbox"]

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
                existing_tracker.update(confidence, frame_num, bbox)

                # Si encontramos una versión mejor del texto, actualizar
                if confidence > existing_tracker.best_confidence:
                    # Mover tracker a la nueva clave si el texto cambió
                    if plate_text != existing_tracker.plate_text:
                        del trackers[existing_tracker.plate_text]
                        existing_tracker.plate_text = plate_text
                        trackers[plate_text] = existing_tracker
            else:
                # Crear nuevo tracker
                new_tracker = PlateTracker(
                    plate_text=plate_text,
                    best_confidence=confidence,
                    best_frame=frame_num,
                    detection_count=1,
                    first_seen=frame_num,
                    last_seen=frame_num
                )
                new_tracker.update(confidence, frame_num, bbox)
                trackers[plate_text] = new_tracker

    def _are_plates_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Verifica si dos textos de placa son similares (para manejar errores OCR)"""
        if not text1 or not text2:
            return False

        # Normalizar textos
        text1 = text1.upper().replace('-', '').replace(' ', '')
        text2 = text2.upper().replace('-', '').replace(' ', '')

        if text1 == text2:
            return True

        # Calcular similitud usando distancia de Levenshtein simplificada
        if len(text1) != len(text2):
            return False

        differences = sum(c1 != c2 for c1, c2 in zip(text1, text2))
        similarity = 1 - (differences / len(text1))

        return similarity >= threshold

    def _extract_unique_plates(self, trackers: Dict[str, PlateTracker]) -> List[Dict[str, Any]]:
        """Extrae placas únicas finales con filtrado y validación"""
        unique_plates = []

        for plate_text, tracker in trackers.items():
            # Filtrar trackers con pocas detecciones (posibles falsos positivos)
            if tracker.detection_count < self.min_detection_frames:
                logger.debug(f"🔍 Placa '{plate_text}' descartada: pocas detecciones ({tracker.detection_count})")
                continue

            # Crear resultado final para esta placa
            plate_result = {
                "plate_text": tracker.plate_text,
                "best_confidence": tracker.best_confidence,
                "average_confidence": sum(tracker.confidences) / len(tracker.confidences),
                "detection_count": tracker.detection_count,
                "first_seen_frame": tracker.first_seen,
                "last_seen_frame": tracker.last_seen,
                "best_frame": tracker.best_frame,
                "duration_frames": tracker.last_seen - tracker.first_seen + 1,
                "average_bbox": tracker.get_average_bbox(),
                "is_valid_format": self._validate_plate_format(tracker.plate_text),
                "stability_score": self._calculate_stability_score(tracker)
            }

            unique_plates.append(plate_result)

        # Ordenar por confianza y estabilidad
        unique_plates.sort(
            key=lambda x: (x["best_confidence"] * 0.7 + x["stability_score"] * 0.3),
            reverse=True
        )

        return unique_plates

    def _validate_plate_format(self, plate_text: str) -> bool:
        """Valida formato de placa peruana"""
        import re
        patterns = [
            r'^[A-Z]{3}-\d{3}$',  # ABC-123
            r'^[A-Z]{2}-\d{4}$',  # AB-1234
            r'^[A-Z]\d{2}-\d{3}$',  # A12-345
        ]

        for pattern in patterns:
            if re.match(pattern, plate_text):
                return True
        return False

    def _calculate_stability_score(self, tracker: PlateTracker) -> float:
        """Calcula score de estabilidad basado en consistencia de detecciones"""
        if not tracker.confidences:
            return 0.0

        # Factores de estabilidad
        confidence_consistency = 1.0 - (np.std(tracker.confidences) / np.mean(tracker.confidences))
        detection_frequency = min(tracker.detection_count / 10.0, 1.0)  # Normalizar a máximo 10

        return (confidence_consistency * 0.6 + detection_frequency * 0.4)

    def _generate_video_result_message(
            self,
            unique_plates: List[Dict[str, Any]],
            tracking_result: Dict[str, Any]
    ) -> str:
        """Genera mensaje descriptivo del resultado del video"""
        if not unique_plates:
            return f"No se detectaron placas válidas en el video. " \
                   f"Frames procesados: {tracking_result['frames_processed']}"

        valid_plates = [p for p in unique_plates if p['is_valid_format']]
        best_plate = unique_plates[0]

        if valid_plates:
            return f"Se detectaron {len(unique_plates)} placa(s) única(s). " \
                   f"Mejor: '{best_plate['plate_text']}' " \
                   f"(Confianza: {best_plate['best_confidence']:.3f}, " \
                   f"Detecciones: {best_plate['detection_count']})"
        else:
            return f"Se detectaron {len(unique_plates)} placa(s) pero con formato no válido. " \
                   f"Mejor: '{best_plate['plate_text']}'"

    async def _save_best_frames(
            self,
            video_path: str,
            unique_plates: List[Dict[str, Any]],
            result_id: str
    ) -> Dict[str, str]:
        """Guarda los frames con las mejores detecciones"""
        frame_urls = {}

        try:
            cap = cv2.VideoCapture(video_path)

            for i, plate in enumerate(unique_plates[:5]):  # Máximo 5 mejores placas
                try:
                    # Ir al frame con mejor detección
                    cap.set(cv2.CAP_PROP_POS_FRAMES, plate['best_frame'])
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

                except Exception as e:
                    logger.warning(f"⚠️ Error guardando frame para placa {plate['plate_text']}: {str(e)}")

            cap.release()

        except Exception as e:
            logger.warning(f"⚠️ Error guardando frames: {str(e)}")

        return frame_urls

    async def _create_annotated_video(
            self,
            video_path: str,
            frame_results: List[Dict[str, Any]],
            result_id: str
    ) -> Optional[str]:
        """Crea video anotado con detecciones (implementación futura)"""
        # TODO: Implementar creación de video anotado
        logger.info("📹 Creación de video anotado pendiente de implementación")
        return None


# Instancia global del servicio
video_service = VideoService()