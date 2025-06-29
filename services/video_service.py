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
    raw_plate_text: str
    best_confidence: float
    best_frame: int
    detection_count: int
    first_seen: int
    last_seen: int
    bbox_history: List[List[float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    is_six_char_valid: bool = False
    auto_formatted: bool = False

    def update(self, confidence: float, frame_num: int, bbox: List[float], is_six_char: bool = False,
               auto_formatted: bool = False):
        """Actualiza el tracker con nueva detección"""
        self.detection_count += 1
        self.last_seen = frame_num
        self.bbox_history.append(bbox)
        self.confidences.append(confidence)

        if is_six_char:
            self.is_six_char_valid = True

        if auto_formatted:
            self.auto_formatted = True

        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_frame = frame_num

    def _calculate_average_bbox(self, bbox_history: List[List[float]]) -> List[float]:
        """Calcula bbox promedio"""
        if not bbox_history:
            return [0, 0, 0, 0]

        avg_bbox = [0.0, 0.0, 0.0, 0.0]
        for bbox in bbox_history:
            for i in range(4):
                avg_bbox[i] += bbox[i]

        for i in range(4):
            avg_bbox[i] /= len(bbox_history)

        return avg_bbox

    def is_similar_position(self, new_bbox: List[float], threshold: float = None) -> bool:
        """Verifica si una nueva detección está en posición similar usando config centralizada"""
        if threshold is None:
            threshold = settings.tracking_iou_threshold

        if not self.bbox_history:
            return False

        last_bbox = self.bbox_history[-1]
        iou = self._calculate_iou(last_bbox, new_bbox)
        return iou > threshold

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calcula Intersection over Union entre dos bboxes"""
        try:
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2

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


class VideoService:
    """Servicio para procesamiento de videos con configuración centralizada"""

    def __init__(self):
        self.model_manager = model_manager
        self.file_service = file_service

        # ✅ INICIALIZAR PIPELINE MEJORADO
        self.enhanced_pipeline = EnhancedALPRPipeline(model_manager)

        # ✅ CONFIGURACIONES CENTRALIZADAS
        self.video_config = settings.get_video_detection_config()
        self.tracking_config = settings.get_tracking_config()
        self.validation_config = settings.get_validation_config()

        # Aplicar configuración centralizada
        self.frame_skip = self.video_config['frame_skip']
        self.similarity_threshold = self.tracking_config['similarity_threshold']
        self.min_detection_frames = self.tracking_config['min_detection_frames']
        self.max_tracking_distance = self.tracking_config['max_tracking_distance']

        # Thread pool para procesamiento paralelo
        self.executor = ThreadPoolExecutor(max_workers=2)

        logger.info("🎬 VideoService inicializado con configuración centralizada")
        logger.debug(f"📊 Config video: {self.video_config}")
        logger.debug(f"🎯 Config tracking: {self.tracking_config}")

    async def process_video(
            self,
            video_path: str,
            file_info: Dict[str, Any],
            request_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa un video completo usando configuración centralizada
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

            # ✅ VERIFICAR DURACIÓN USANDO CONFIG CENTRALIZADA
            max_duration = request_params.get('max_duration', self.video_config['max_duration'])
            if video_info['duration'] > max_duration:
                raise Exception(f"Video muy largo. Máximo: {max_duration}s, "
                                f"recibido: {video_info['duration']:.1f}s")

            # ✅ CONFIGURAR PARÁMETROS USANDO CONFIG CENTRALIZADA
            final_params = self._merge_params_with_config(request_params)
            self.frame_skip = final_params['frame_skip']

            logger.info(f"⚙️ Parámetros finales: confidence={final_params['confidence_threshold']}, "
                        f"frame_skip={final_params['frame_skip']}, "
                        f"min_detection_frames={final_params['min_detection_frames']}")

            # ✅ PROCESAR CON ROI Y FILTRO DE 6 CARACTERES SIN GUIÓN
            with PerformanceTimer("Procesamiento de video con configuración centralizada"):
                tracking_result = await self._process_video_with_roi_and_filter(
                    video_path, video_info, final_params
                )
                unique_plates = self._extract_unique_plates_enhanced(tracking_result['trackers'])

            # ✅ GUARDAR RESULTADOS USANDO CONFIG CENTRALIZADA
            result_urls = {}
            save_results = final_params.get('save_results', self.video_config['save_results'])

            if save_results:
                try:
                    # Copiar video original
                    original_path = await self.file_service.copy_to_results(
                        video_path, result_id, "original"
                    )
                    result_urls["original"] = self.file_service.get_file_url(original_path)

                    # Guardar frames con mejores detecciones
                    save_best_frames = final_params.get('save_best_frames', self.video_config['save_best_frames'])
                    if save_best_frames:
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
                    "six_char_plates": len([p for p in unique_plates if p.get('is_six_char_valid', False)]),
                    "auto_formatted_plates": len([p for p in unique_plates if p.get('auto_formatted', False)]),
                    "configuration_used": "centralized_settings"  # ✅ NUEVO
                },
                "unique_plates": unique_plates,
                "best_plate": unique_plates[0] if unique_plates else None,
                "processing_time": round(processing_time, 3),
                "timestamp": time.time(),
                "result_urls": result_urls if result_urls else None,
                "tracking_type": "enhanced_roi_6chars_centralized",  # ✅ ACTUALIZADO
                "enhancement_info": {
                    "roi_enabled": settings.roi_enabled,
                    "six_char_filter": settings.force_six_characters,
                    "roi_percentage": settings.roi_percentage,
                    "model_expects_dash": False,
                    "auto_dash_formatting": True,
                    "configuration_source": "centralized_settings"  # ✅ NUEVO
                },
                "config_info": {  # ✅ INFORMACIÓN DE CONFIG USADA
                    "final_params": final_params,
                    "video_config": self.video_config,
                    "tracking_config": self.tracking_config,
                    "source": "settings.py + .env"
                }
            }

            # Limpiar archivo temporal
            self.file_service.cleanup_temp_file(video_path)

            # ✅ LOG MEJORADO CON INFO DE CONFIG
            six_char_count = len([p for p in unique_plates if p.get('is_six_char_valid', False)])
            auto_formatted_count = len([p for p in unique_plates if p.get('auto_formatted', False)])

            logger.success(f"✅ Video procesado en {processing_time:.3f}s con config centralizada. "
                           f"Placas únicas: {len(unique_plates)} "
                           f"(6 chars válidas: {six_char_count}, "
                           f"auto-formateadas: {auto_formatted_count})")

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
                    "six_char_plates": 0,
                    "auto_formatted_plates": 0,
                    "configuration_used": "error"
                },
                "unique_plates": [],
                "best_plate": None,
                "processing_time": round(processing_time, 3),
                "timestamp": time.time(),
                "result_urls": None,
                "tracking_type": "error"
            }

    def _merge_params_with_config(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """✅ NUEVO: Combina parámetros de request con configuración centralizada"""

        # Usar config centralizada como base
        final_params = self.video_config.copy()

        # Sobrescribir con parámetros específicos del request (si se proporcionan)
        param_mappings = {
            'confidence_threshold': 'confidence_threshold',
            'iou_threshold': 'iou_threshold',
            'frame_skip': 'frame_skip',
            'max_duration': 'max_duration',
            'min_detection_frames': 'min_detection_frames',
            'save_results': 'save_results',
            'save_best_frames': 'save_best_frames',
            'create_annotated_video': 'create_annotated_video'
        }

        for request_key, config_key in param_mappings.items():
            if request_key in request_params:
                final_params[config_key] = request_params[request_key]
                logger.debug(f"🔧 Sobrescribiendo {config_key}: {request_params[request_key]} (desde request)")

        # Agregar configuraciones de tracking
        final_params.update({
            'similarity_threshold': self.tracking_config['similarity_threshold'],
            'max_tracking_distance': self.tracking_config['max_tracking_distance'],
            'tracking_iou_threshold': self.tracking_config['iou_threshold']
        })

        logger.debug(f"📋 Parámetros finales combinados: {final_params}")

        return final_params

    async def _process_video_with_roi_and_filter(
            self,
            video_path: str,
            video_info: Dict[str, Any],
            final_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """✅ ACTUALIZADO: Procesa video usando configuración centralizada"""

        plate_trackers: Dict[str, PlateTracker] = {}
        frame_results = []

        frames_processed = 0
        frames_with_detections = 0
        total_detections = 0
        six_char_detections = 0
        auto_formatted_detections = 0

        cap = cv2.VideoCapture(video_path)
        frame_num = 0

        # ✅ USAR PARÁMETROS CENTRALIZADOS
        confidence_threshold = final_params['confidence_threshold']
        iou_threshold = final_params['iou_threshold']
        frame_skip = final_params['frame_skip']

        logger.info(f"⚙️ Config centralizada aplicada: confidence={confidence_threshold}, "
                    f"iou={iou_threshold}, frame_skip={frame_skip}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Procesar solo cada N frames (usando config centralizada)
                if frame_num % frame_skip == 0:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # ✅ PROCESAR CON PIPELINE MEJORADO
                        frame_result = self._process_single_frame_with_enhancements(
                            frame_rgb, frame_num, confidence_threshold, iou_threshold
                        )

                        frames_processed += 1

                        if frame_result['detections']:
                            frames_with_detections += 1
                            total_detections += len(frame_result['detections'])

                            six_char_count = len(
                                [d for d in frame_result['detections'] if d.get('six_char_validated', False)])
                            auto_formatted_count = len(
                                [d for d in frame_result['detections'] if d.get('auto_formatted', False)])

                            six_char_detections += six_char_count
                            auto_formatted_detections += auto_formatted_count

                            logger.debug(f"🎯 Frame {frame_num}: {len(frame_result['detections'])} detecciones "
                                         f"({six_char_count} válidas 6chars, {auto_formatted_count} auto-formateadas)")

                            # ✅ ACTUALIZAR TRACKERS CON CONFIG CENTRALIZADA
                            self._update_trackers_enhanced(
                                plate_trackers,
                                frame_result['detections'],
                                frame_num,
                                final_params
                            )

                        frame_results.append(frame_result)

                        # Log de progreso cada 30 frames procesados
                        if frames_processed % 30 == 0:
                            progress = (frame_num / video_info['total_frames']) * 100
                            six_char_plates = len([t for t in plate_trackers.values() if t.is_six_char_valid])
                            auto_formatted_plates = len([t for t in plate_trackers.values() if t.auto_formatted])

                            logger.info(f"📊 Progreso: {progress:.1f}% - "
                                        f"Placas únicas: {len(plate_trackers)} "
                                        f"(6 chars: {six_char_plates}, auto: {auto_formatted_plates})")

                    except Exception as e:
                        logger.warning(f"⚠️ Error procesando frame {frame_num}: {str(e)}")

                frame_num += 1

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
            "six_char_detections": six_char_detections,
            "auto_formatted_detections": auto_formatted_detections,
            "config_used": final_params  # ✅ NUEVO
        }

    def _process_single_frame_with_enhancements(
            self,
            frame: np.ndarray,
            frame_num: int,
            confidence_threshold: float,
            iou_threshold: float
    ) -> Dict[str, Any]:
        """✅ ACTUALIZADO: Procesa frame usando configuración centralizada"""
        try:
            # ✅ USAR ROI Y FILTROS SEGÚN CONFIG CENTRALIZADA
            result = self.enhanced_pipeline.process_with_enhancements(
                frame,
                use_roi=settings.roi_enabled,
                filter_six_chars=False,
                return_stats=False,
                conf=confidence_threshold,
                iou=iou_threshold
            )

            detections = []
            if result.get("success") and result.get("final_results"):
                for plate_result in result["final_results"]:
                    formatted_text = plate_result.get("plate_text", "")
                    raw_text = plate_result.get("raw_plate_text", "")

                    if formatted_text or raw_text:
                        detection = {
                            "plate_text": formatted_text,
                            "raw_plate_text": raw_text,
                            "confidence": plate_result["overall_confidence"],
                            "plate_bbox": plate_result["plate_bbox"],
                            "plate_confidence": plate_result["plate_confidence"],
                            "char_confidence": plate_result.get("character_recognition", {}).get("confidence", 0.0),
                            "is_valid_plate": plate_result["is_valid_plate"],
                            "six_char_validated": plate_result.get("six_char_validated", False),
                            "auto_formatted": plate_result.get("auto_formatted", False),
                            "validation_info": plate_result.get("validation_info", {}),
                            "frame_num": frame_num,
                            "processing_method": "roi_6chars_centralized"  # ✅ ACTUALIZADO
                        }
                        detections.append(detection)

                        logger.debug(f"✅ [FRAME {frame_num}] Placa: '{raw_text}' -> '{formatted_text}' "
                                     f"(6chars: {detection['six_char_validated']}, auto: {detection['auto_formatted']})")

            return {
                "frame_num": frame_num,
                "detections": detections,
                "processing_success": result.get("success", False),
                "roi_used": result.get("use_roi", settings.roi_enabled),
                "filter_applied": result.get("filter_six_chars", settings.force_six_characters),
                "model_info": result.get("model_info", {}),
                "config_source": "centralized_settings"  # ✅ NUEVO
            }

        except Exception as e:
            logger.warning(f"⚠️ Error en frame {frame_num}: {str(e)}")
            return {
                "frame_num": frame_num,
                "detections": [],
                "processing_success": False,
                "error": str(e),
                "roi_used": settings.roi_enabled,
                "filter_applied": settings.force_six_characters,
                "config_source": "centralized_settings"
            }

    def _update_trackers_enhanced(self, trackers: Dict[str, PlateTracker],
                                  detections: List[Dict[str, Any]], frame_num: int,
                                  final_params: Dict[str, Any]):
        """Tracking espacial para múltiples placas"""

        #  UMBRALES MENOS AGRESIVOS
        similarity_threshold = final_params.get('similarity_threshold', 0.6)  # Era: 0.8
        spatial_threshold = final_params.get('tracking_iou_threshold', 0.15)  # Era: 0.2

        logger.debug(f"🔄 Procesando {len(detections)} detecciones en frame {frame_num} "
                     f"(sim_thresh={similarity_threshold}, spatial_thresh={spatial_threshold})")

        for detection in detections:
            formatted_text = detection["plate_text"]
            raw_text = detection.get("raw_plate_text", "")
            confidence = detection["confidence"]
            bbox = detection["plate_bbox"]
            is_six_char = detection.get("six_char_validated", False)
            auto_formatted = detection.get("auto_formatted", False)

            # CREAR clave única basada en posición espacial
            bbox_region = self._get_bbox_region(bbox, 1920, 1080)
            tracking_key = f"{formatted_text}_{bbox_region}" if formatted_text else f"{raw_text}_{bbox_region}"

            # Buscar tracker existente por proximidad espacial primero
            existing_tracker = None

            # 1. Buscar por clave exacta
            if tracking_key in trackers:
                existing_tracker = trackers[tracking_key]
                logger.debug(f"🔗 Tracker exacto encontrado: {tracking_key}")
            else:
                # 2. Buscar por proximidad espacial + similitud de texto
                for tracked_key, tracker in trackers.items():
                    text_similar = self._are_plates_similar(
                        formatted_text or raw_text,
                        tracker.plate_text,
                        similarity_threshold
                    )
                    spatial_similar = tracker.is_similar_position(bbox, spatial_threshold)

                    if text_similar and spatial_similar:
                        existing_tracker = tracker
                        logger.debug(f"🔗 Fusionando con tracker similar: {tracked_key} "
                                     f"(text_sim={text_similar}, spatial_sim={spatial_similar})")
                        break

            if existing_tracker:
                existing_tracker.update(confidence, frame_num, bbox, is_six_char, auto_formatted)
                logger.debug(f"✅ Tracker actualizado: {existing_tracker.plate_text} "
                             f"(detecciones: {existing_tracker.detection_count})")
            else:
                # CREAR NUEVO TRACKER (menos restrictivo)
                new_tracker = PlateTracker(
                    plate_text=formatted_text,
                    raw_plate_text=raw_text,
                    best_confidence=confidence,
                    best_frame=frame_num,
                    detection_count=1,
                    first_seen=frame_num,
                    last_seen=frame_num,
                    is_six_char_valid=is_six_char,
                    auto_formatted=auto_formatted
                )
                new_tracker.update(confidence, frame_num, bbox, is_six_char, auto_formatted)
                trackers[tracking_key] = new_tracker

                logger.debug(f"🆕 Nuevo tracker creado: '{raw_text}' -> '{formatted_text}' "
                             f"en región {bbox_region}")

    def _extract_unique_plates_enhanced(self, trackers: Dict[str, PlateTracker]) -> List[Dict[str, Any]]:
        """Extraer TODAS las placas válidas por región espacial"""

        unique_plates = []
        min_frames = self.tracking_config['min_detection_frames']  # Ahora: 2

        logger.info(f"📊 Extrayendo placas únicas: {len(trackers)} trackers, min_frames={min_frames}")

        # AGRUPAR por regiones espaciales para evitar duplicados
        spatial_groups = defaultdict(list)

        for plate_text, tracker in trackers.items():
            if tracker.detection_count >= min_frames:
                # Extraer región espacial de la clave
                parts = plate_text.split('_')
                spatial_region = parts[-1] if len(parts) > 1 and parts[-1].startswith('R') else "R0_0"
                spatial_groups[spatial_region].append(tracker)

        logger.debug(f"🗂️ Agrupación espacial: {len(spatial_groups)} regiones")

        # PROCESAR CADA REGIÓN ESPACIAL
        for spatial_region, region_trackers in spatial_groups.items():
            # Ordenar por confianza en esta región
            region_trackers.sort(key=lambda t: t.best_confidence, reverse=True)

            logger.debug(f"📍 Región {spatial_region}: {len(region_trackers)} trackers")

            # PROCESAR TODAS las placas válidas de esta región
            for i, tracker in enumerate(region_trackers):
                avg_confidence = sum(tracker.confidences) / len(tracker.confidences) if tracker.confidences else 0.0
                stability_score = 1.0 - (np.std(tracker.confidences) / np.mean(tracker.confidences)) if len(
                    tracker.confidences) > 1 and np.mean(tracker.confidences) > 0 else 0.5

                plate_result = {
                    "plate_text": tracker.plate_text,
                    "raw_plate_text": tracker.raw_plate_text,
                    "best_confidence": tracker.best_confidence,
                    "detection_count": tracker.detection_count,
                    "first_seen_frame": tracker.first_seen,
                    "last_seen_frame": tracker.last_seen,
                    "best_frame": tracker.best_frame,
                    "is_valid_format": self._validate_plate_format(tracker.plate_text),
                    "is_six_char_valid": tracker.is_six_char_valid,
                    "auto_formatted": tracker.auto_formatted,
                    "avg_confidence": round(avg_confidence, 3),
                    "stability_score": round(stability_score, 3),
                    "duration_frames": tracker.last_seen - tracker.first_seen + 1,
                    "char_count": len(tracker.raw_plate_text) if tracker.raw_plate_text else 0,
                    "spatial_region": spatial_region,  # NUEVO
                    "avg_bbox": self._calculate_average_bbox(tracker.bbox_history),  # NUEVO
                    "region_rank": i + 1,  #   en esta región
                    "processing_method": "spatial_tracking_multiple_plates",
                    "config_info": {
                        "min_frames_required": min_frames,
                        "frames_detected": tracker.detection_count,
                        "meets_requirements": tracker.detection_count >= min_frames,
                        "spatial_region": spatial_region
                    }
                }
                unique_plates.append(plate_result)

                status_indicator = "✅" if tracker.is_six_char_valid else "❌"
                auto_indicator = "🔧" if tracker.auto_formatted else ""

                logger.info(f"📋 Placa confirmada: '{tracker.raw_plate_text}' -> '{tracker.plate_text}' "
                            f"{status_indicator}{auto_indicator} "
                            f"(Región: {spatial_region}, Detecciones: {tracker.detection_count}/{min_frames}, "
                            f"Confianza: {tracker.best_confidence:.3f})")

        # ORDENAR POR PRIORIDAD: 1) 6 chars válidas, 2) Confianza
        unique_plates.sort(key=lambda x: (x["is_six_char_valid"], x["best_confidence"]), reverse=True)

        logger.success(f"📊 Extracción completada: {len(unique_plates)} placas únicas "
                       f"en {len(spatial_groups)} regiones espaciales")

        return unique_plates  # RETORNAR TODAS, sin límite artificial

    def _get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """✅ ACTUALIZADO: Obtiene información del video usando config centralizada"""
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

            # ✅ USAR FRAME_SKIP CENTRALIZADO
            frame_skip = self.video_config['frame_skip']

            return {
                "fps": fps,
                "total_frames": frame_count,
                "width": width,
                "height": height,
                "duration": duration,
                "frames_to_process": frame_count // frame_skip,
                "file_size_mb": self.file_service.get_file_size_mb(video_path),
                "config_frame_skip": frame_skip  # ✅ NUEVO
            }

        except Exception as e:
            logger.error(f"❌ Error obteniendo info del video: {str(e)}")
            return None

    def _are_plates_similar(self, text1: str, text2: str, threshold: float = None) -> bool:
        """✅ ACTUALIZADO: Verifica similitud usando umbral centralizado"""
        if threshold is None:
            threshold = self.tracking_config['similarity_threshold']

        if not text1 or not text2:
            return False

        text1 = text1.upper().replace('-', '').replace(' ', '')
        text2 = text2.upper().replace('-', '').replace(' ', '')

        if text1 == text2:
            return True

        if len(text1) != len(text2):
            return False

        differences = sum(c1 != c2 for c1, c2 in zip(text1, text2))
        similarity = 1 - (differences / len(text1))

        return similarity >= threshold

    def _validate_plate_format(self, plate_text: str) -> bool:
        """✅ CORREGIDO: Valida formato de placa peruana (con guión)"""
        import re

        if not plate_text:
            return False

        patterns = [
            r'^[A-Z]{3}-\d{3}$',  # ABC-123
            r'^[A-Z]{2}-\d{4}$',  # AB-1234
            r'^[A-Z]\d{2}-\d{3}$',  # A12-345
        ]

        for pattern in patterns:
            if re.match(pattern, plate_text):
                return True

        return False

    def _get_is_valid(self, plate: Dict[str, Any]) -> bool:
        """Obtiene validez de placa según el tipo de resultado"""
        return plate.get("is_valid_format", False)

    def _generate_video_result_message(self, unique_plates: List[Dict[str, Any]],
                                       tracking_result: Dict[str, Any]) -> str:
        """Genera mensaje con información de múltiples placas"""
        if not unique_plates:
            return f"No se detectaron placas válidas en el video (config centralizada). " \
                   f"Frames procesados: {tracking_result['frames_processed']}, " \
                   f"detecciones totales: {tracking_result['total_detections']}"

        valid_plates = [p for p in unique_plates if p['is_valid_format']]
        six_char_plates = [p for p in unique_plates if p.get('is_six_char_valid', False)]
        auto_formatted_plates = [p for p in unique_plates if p.get('auto_formatted', False)]
        spatial_regions = len(set(p.get('spatial_region', 'R0_0') for p in unique_plates))
        best_plate = unique_plates[0]

        message = f"Se detectaron {len(unique_plates)} placa(s) única(s) en {spatial_regions} regiones espaciales. "

        if six_char_plates:
            message += f"{len(six_char_plates)} con 6 caracteres válidos. "

        if auto_formatted_plates:
            message += f"{len(auto_formatted_plates)} auto-formateadas. "

        raw_text = best_plate.get('raw_plate_text', '')
        formatted_text = best_plate['plate_text']

        if raw_text and raw_text != formatted_text:
            message += f"Mejor: '{raw_text}' → '{formatted_text}' "
        else:
            message += f"Mejor: '{formatted_text}' "

        message += f"(Región: {best_plate.get('spatial_region', 'R0_0')}, " \
                   f"Confianza: {best_plate['best_confidence']:.3f}, " \
                   f"Detecciones: {best_plate['detection_count']}"

        if best_plate.get('is_six_char_valid'):
            message += ", ✅ 6 chars"

        if best_plate.get('auto_formatted'):
            message += ", 🔧 auto"

        message += ")"

        return message

    async def _save_best_frames(
            self,
            video_path: str,
            unique_plates: List[Dict[str, Any]],
            result_id: str
    ) -> Dict[str, str]:
        """✅ ACTUALIZADO: Guarda frames con configuración centralizada"""
        frame_urls = {}

        try:
            cap = cv2.VideoCapture(video_path)

            for i, plate in enumerate(unique_plates[:5]):
                try:
                    best_frame_num = plate.get('best_frame', 0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_num)
                    ret, frame = cap.read()

                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        mock_detection = [{
                            "plate_bbox": [100, 100, 200, 150],
                            "plate_text": plate["plate_text"],
                            "raw_plate_text": plate.get("raw_plate_text", ""),
                            "six_char_validated": plate.get("is_six_char_valid", False),
                            "auto_formatted": plate.get("auto_formatted", False),
                            "overall_confidence": plate["best_confidence"]
                        }]

                        visualized_frame = self._create_frame_visualization(frame_rgb, mock_detection)

                        raw_text = plate.get('raw_plate_text', plate['plate_text'])
                        safe_filename = raw_text.replace('-', '_').replace(' ', '_')

                        frame_path = await self.file_service.save_result_image(
                            visualized_frame,
                            result_id,
                            f"best_frame_{i + 1}_{safe_filename}_centralized"  # ✅ MARCADOR
                        )

                        frame_urls[f"best_frame_{i + 1}"] = self.file_service.get_file_url(frame_path)
                        logger.info(
                            f"💾 Frame visualizado guardado (config centralizada): '{raw_text}' -> '{plate['plate_text']}'")

                except Exception as e:
                    logger.warning(f"⚠️ Error guardando frame visualizado: {str(e)}")

            cap.release()

        except Exception as e:
            logger.warning(f"⚠️ Error guardando frames visualizados: {str(e)}")

        return frame_urls

    def _create_frame_visualization(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """✅ ACTUALIZADO: Crea visualización usando configuración centralizada"""
        try:
            mock_result = {
                "use_roi": settings.roi_enabled,
                "roi_coords": self._calculate_roi_coords(frame.shape),
                "final_results": detections
            }

            return self.enhanced_pipeline.create_visualization(frame, mock_result)

        except Exception as e:
            logger.debug(f"Error creando visualización: {e}")
            return frame

    def _calculate_roi_coords(self, image_shape) -> Dict[str, int]:
        """✅ ACTUALIZADO: Calcula ROI usando config centralizada"""
        height, width = image_shape[:2]
        roi_percentage = settings.roi_percentage / 100.0  # Convertir a decimal
        roi_width = int(width * roi_percentage)
        roi_height = int(height * roi_percentage)

        return {
            "x_start": (width - roi_width) // 2,
            "y_start": (height - roi_height) // 2,
            "x_end": (width + roi_width) // 2,
            "y_end": (height + roi_height) // 2,
            "width": roi_width,
            "height": roi_height
        }

    # ✅ NUEVOS MÉTODOS PARA CONFIGURACIÓN DINÁMICA

    def get_current_config(self) -> Dict[str, Any]:
        """Obtiene la configuración actual del servicio"""
        return {
            "video_detection": self.video_config,
            "tracking": self.tracking_config,
            "validation": self.validation_config,
            "roi": settings.get_roi_config(),
            "service_state": {
                "frame_skip": self.frame_skip,
                "similarity_threshold": self.similarity_threshold,
                "min_detection_frames": self.min_detection_frames,
                "max_tracking_distance": self.max_tracking_distance
            }
        }

    def update_config_from_settings(self):
        """Recarga la configuración desde settings"""
        self.video_config = settings.get_video_detection_config()
        self.tracking_config = settings.get_tracking_config()
        self.validation_config = settings.get_validation_config()

        # Actualizar variables locales
        self.frame_skip = self.video_config['frame_skip']
        self.similarity_threshold = self.tracking_config['similarity_threshold']
        self.min_detection_frames = self.tracking_config['min_detection_frames']
        self.max_tracking_distance = self.tracking_config['max_tracking_distance']

        logger.info("🔄 Configuración del VideoService recargada desde settings")

    def get_recommended_params_for_video_type(self, video_type: str = "standard") -> Dict[str, Any]:
        """✅ NUEVO: Obtiene parámetros recomendados según el tipo de video"""

        base_config = self.video_config.copy()

        video_type_configs = {
            "standard": base_config,
            "quick": {
                **base_config,
                "confidence_threshold": settings.quick_confidence_threshold,
                "frame_skip": settings.quick_frame_skip,
                "max_duration": settings.quick_max_duration,
                "save_results": False,
                "save_best_frames": False
            },
            "high_precision": {
                **base_config,
                "confidence_threshold": settings.high_confidence_warning,
                "frame_skip": max(1, settings.video_frame_skip - 1),
                "min_detection_frames": settings.stability_frames_required
            },
            "long_video": {
                **base_config,
                "frame_skip": max(5, settings.video_frame_skip + 2),
                "confidence_threshold": settings.video_confidence_threshold + 0.1,
                "create_annotated_video": False
            }
        }

        return video_type_configs.get(video_type, base_config)

    def _get_bbox_region(self, bbox: List[float], image_width: int = 1920, image_height: int = 1080) -> str:
        """Identifica región espacial para evitar fusión de placas diferentes"""
        try:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Dividir imagen en cuadrícula 4x4
            region_x = int(center_x // (image_width / 4))
            region_y = int(center_y // (image_height / 4))

            return f"R{region_x}_{region_y}"
        except Exception:
            return "R0_0"

    def _calculate_average_bbox(self, bbox_history: List[List[float]]) -> List[float]:
        """Calcula bbox promedio de un tracker"""
        if not bbox_history:
            return [0, 0, 0, 0]

        avg_bbox = [0.0, 0.0, 0.0, 0.0]
        for bbox in bbox_history:
            if len(bbox) == 4:  # Verificar que el bbox es válido
                for i in range(4):
                    avg_bbox[i] += bbox[i]

        count = len(bbox_history)
        if count > 0:
            for i in range(4):
                avg_bbox[i] /= count

        return avg_bbox


# Instancia global del servicio
video_service = VideoService()