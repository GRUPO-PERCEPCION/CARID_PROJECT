import time
from typing import Dict, Any, Optional, List
import numpy as np
from loguru import logger

from models.model_manager import model_manager
from services.file_service import file_service
from core.utils import PerformanceTimer
from config.settings import settings


class DetectionService:
    """Servicio principal para detección y reconocimiento de placas con configuración centralizada"""

    def __init__(self):
        self.model_manager = model_manager
        self.file_service = file_service

        # Obtener configuraciones centralizadas
        self.image_config = settings.get_image_detection_config()
        self.validation_config = settings.get_validation_config()

        logger.info("🔍 DetectionService inicializado con configuración centralizada")
        logger.debug(f"📊 Config imágenes: {self.image_config}")

    async def process_image(self, file_path: str, file_info: Dict[str, Any],
                            request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa imagen con detección múltiple mejorada"""
        start_time = time.time()
        result_id = self.file_service.create_result_id()

        try:
            logger.info(f"🔄 Iniciando procesamiento de imagen: {file_info['filename']}")

            # Verificar que los modelos estén cargados
            if not self.model_manager.is_loaded:
                raise Exception("Los modelos no están cargados")

            # APLICAR CONFIGURACIÓN CENTRALIZADA CON FALLBACKS
            model_kwargs = self._build_model_kwargs(request_params)

            logger.debug(f"⚙️ Parámetros finales: {model_kwargs}")

            # Procesar con el pipeline completo
            with PerformanceTimer("Pipeline completo"):
                pipeline_result = self.model_manager.process_full_pipeline(
                    file_path,
                    **model_kwargs
                )

            # LOG DETALLADO de detecciones múltiples
            if pipeline_result.get("success") and pipeline_result.get("final_results"):
                logger.info(f"📊 Detecciones encontradas: {len(pipeline_result['final_results'])} placas")
                for i, result in enumerate(pipeline_result["final_results"][:10]):  # Log primeras 10
                    logger.debug(f"  Placa {i + 1}: '{result.get('plate_text', 'N/A')}' "
                                 f"(conf: {result.get('overall_confidence', 0):.3f})")

            # Guardar imágenes si se solicita (usando config centralizada)
            result_urls = {}
            save_results = request_params.get('save_results', self.image_config['save_results'])

            if save_results and pipeline_result.get("success"):
                try:
                    # Copiar imagen original
                    original_path = await self.file_service.copy_to_results(
                        file_path, result_id, "original"
                    )
                    result_urls["original"] = self.file_service.get_file_url(original_path)

                    # Generar visualización si se solicita (usando config centralizada)
                    return_viz = request_params.get('return_visualization',
                                                    self.image_config['return_visualization'])
                    if return_viz:
                        visualization = await self._create_visualization(
                            file_path, pipeline_result, result_id
                        )
                        if visualization:
                            result_urls["visualization"] = visualization

                except Exception as e:
                    logger.warning(f"⚠️ Error guardando resultados: {str(e)}")

            # Tiempo de procesamiento
            processing_time = time.time() - start_time

            # Crear resultado final con estructura compatible
            processed_plates = []
            if pipeline_result.get("success") and pipeline_result.get("final_results"):
                for result in pipeline_result["final_results"]:
                    processed_plate = {
                        "plate_id": result["plate_id"],
                        "plate_bbox": result["plate_bbox"],
                        "plate_confidence": result["plate_confidence"],
                        "plate_area": result["plate_area"],
                        "character_recognition": result["character_recognition"],
                        "plate_text": result["plate_text"],
                        "overall_confidence": result["overall_confidence"],
                        "is_valid_plate": result["is_valid_plate"],
                        "spatial_region": self._get_plate_region(result["plate_bbox"])  # NUEVA
                    }
                    processed_plates.append(processed_plate)

            # ANÁLISIS ESPACIAL DE LAS DETECCIONES
            spatial_regions = len(set(p["spatial_region"] for p in processed_plates))
            confidence_breakdown = {
                "high": len([p for p in processed_plates if p["overall_confidence"] > 0.7]),
                "medium": len([p for p in processed_plates if 0.4 <= p["overall_confidence"] <= 0.7]),
                "low": len([p for p in processed_plates if p["overall_confidence"] < 0.4])
            }

            # Crear resultado final
            result = {
                "success": pipeline_result.get("success", False),
                "message": self._generate_result_message(pipeline_result, processed_plates),
                "file_info": file_info,
                "plates_processed": len(processed_plates),
                "plate_detection": pipeline_result.get("plate_detection"),
                "final_results": processed_plates,
                "best_result": processed_plates[0] if processed_plates else None,
                "all_detected_plates": processed_plates,  # Lista completa
                "processing_summary": {
                    "plates_detected": pipeline_result.get("plates_processed", 0),
                    "plates_with_text": len([p for p in processed_plates if p["plate_text"]]),
                    "valid_plates": len([p for p in processed_plates if p["is_valid_plate"]]),
                    "total_plates_found": len(processed_plates),  # NUEVA
                    "spatial_regions_covered": spatial_regions,  # NUEVA
                    "confidence_distribution": confidence_breakdown,  # NUEVA
                    "configuration_used": "centralized_settings"
                },
                "plates_by_confidence": confidence_breakdown,  # NUEVA
                "spatial_distribution": {  # NUEVA
                    region: len([p for p in processed_plates if p["spatial_region"] == region])
                    for region in set(p["spatial_region"] for p in processed_plates)
                },
                "processing_time": round(processing_time, 3),
                "timestamp": time.time(),
                "result_urls": result_urls if result_urls else None,
                "config_info": {  # INFORMACIÓN DE CONFIGURACIÓN USADA
                    "confidence_threshold": model_kwargs['conf'],
                    "iou_threshold": model_kwargs['iou'],
                    "max_detections": request_params.get('max_detections', self.image_config['max_detections']),
                    "source": "centralized_settings"
                }
            }

            # Limpiar archivo temporal
            self.file_service.cleanup_temp_file(file_path)

            logger.success(f"✅ Procesamiento completado en {processing_time:.3f}s - "
                           f"{len(processed_plates)} placas en {spatial_regions} regiones")
            return result

        except Exception as e:
            logger.error(f"❌ Error procesando imagen: {str(e)}")

            # Limpiar archivo temporal en caso de error
            self.file_service.cleanup_temp_file(file_path)

            processing_time = time.time() - start_time

            # Retornar resultado de error
            return {
                "success": False,
                "message": f"Error procesando imagen: {str(e)}",
                "file_info": file_info,
                "plates_processed": 0,
                "plate_detection": None,
                "final_results": [],
                "best_result": None,
                "all_detected_plates": [],  # NUEVA
                "processing_summary": {
                    "plates_detected": 0,
                    "plates_with_text": 0,
                    "valid_plates": 0,
                    "total_plates_found": 0,
                    "spatial_regions_covered": 0,
                    "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
                    "configuration_used": "error"
                },
                "plates_by_confidence": {"high": 0, "medium": 0, "low": 0},
                "spatial_distribution": {},
                "processing_time": round(processing_time, 3),
                "timestamp": time.time(),
                "result_urls": None
            }

    def _build_model_kwargs(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """ACTUALIZADO: Construye parámetros del modelo para detección múltiple"""

        # OBTENER max_detections más alto
        max_detections = request_params.get('max_detections', self.image_config['max_detections'])

        model_kwargs = {
            'conf': request_params.get('confidence_threshold', self.image_config['confidence_threshold']),
            'iou': request_params.get('iou_threshold', self.image_config['iou_threshold']),
            'max_det': max_detections,  # ✅ CRÍTICO: Pasar al modelo
            'verbose': False
        }

        # LOG para debugging múltiples detecciones
        logger.info(f"🎯 Detección múltiple habilitada: max_det={max_detections}, "
                    f"conf={model_kwargs['conf']}, iou={model_kwargs['iou']}")

        logger.debug(f"🔧 Config base: {self.image_config}")
        logger.debug(f"📝 Request params: {request_params}")
        logger.debug(f"⚙️ Kwargs finales: {model_kwargs}")

        return model_kwargs

    def _get_plate_region(self, bbox: List[float]) -> str:
        """Obtiene región espacial de una placa"""
        try:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Dividir imagen en regiones de 400x300px aproximadamente
            region_x = int(center_x // 400)
            region_y = int(center_y // 300)

            return f"R{region_x}_{region_y}"
        except Exception:
            return "R0_0"

    def _generate_result_message(self, pipeline_result: Dict[str, Any], processed_plates: List[Dict]) -> str:
        """Genera mensaje descriptivo del resultado"""
        if not pipeline_result.get("success"):
            return "No se detectaron placas en la imagen"

        if not processed_plates:
            return "Se detectaron regiones de placas pero no se pudo reconocer texto"

        valid_plates = [p for p in processed_plates if p["is_valid_plate"]]

        if valid_plates:
            best_plate = valid_plates[0]
            return f"Se detectó placa válida: '{best_plate['plate_text']}' con confianza {best_plate['overall_confidence']:.2f}"
        else:
            best_plate = processed_plates[0]
            return f"Se detectó texto: '{best_plate['plate_text']}' pero formato no válido"

    async def _create_visualization(self, file_path: str, pipeline_result: Dict[str, Any], result_id: str) -> Optional[
        str]:
        """Crea una imagen de visualización con las detecciones"""
        try:
            # Crear visualización de detección de placas
            if pipeline_result.get("plate_detection") and pipeline_result["plate_detection"]["success"]:
                plate_viz = self.model_manager.plate_detector.visualize_detections(file_path)

                # Guardar visualización
                viz_path = await self.file_service.save_result_image(
                    plate_viz, result_id, "visualization"
                )

                return self.file_service.get_file_url(viz_path)

            return None

        except Exception as e:
            logger.warning(f"⚠️ Error creando visualización: {str(e)}")
            return None

    async def get_processing_stats(self) -> Dict[str, Any]:
        """✅ ACTUALIZADO: Obtiene estadísticas del servicio incluyendo configuración"""
        try:
            # Información de modelos
            model_info = self.model_manager.get_model_info()

            # Información de archivos
            temp_files = list(self.file_service.temp_dir.glob("*"))
            result_files = list(self.file_service.results_dir.glob("*"))

            return {
                "models_status": {
                    "loaded": model_info["models_loaded"],
                    "device": model_info["device"],
                    "plate_detector": model_info["plate_detector_loaded"],
                    "char_recognizer": model_info["char_recognizer_loaded"]
                },
                "file_system": {
                    "temp_files_count": len(temp_files),
                    "result_files_count": len(result_files),
                    "temp_dir_size_mb": sum(f.stat().st_size for f in temp_files if f.is_file()) / (1024 * 1024),
                    "results_dir_size_mb": sum(f.stat().st_size for f in result_files if f.is_file()) / (1024 * 1024)
                },
                "centralized_configuration": {  # ✅ NUEVA SECCIÓN
                    "image_config": self.image_config,
                    "validation_config": self.validation_config,
                    "max_file_size_mb": settings.max_file_size,
                    "roi_enabled": settings.roi_enabled,
                    "force_six_characters": settings.force_six_characters,
                    "configuration_source": "settings.py + .env"
                },
                "legacy_configuration": {  # ✅ PARA COMPATIBILIDAD
                    "confidence_threshold": self.image_config['confidence_threshold'],
                    "iou_threshold": self.image_config['iou_threshold']
                }
            }

        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {str(e)}")
            return {"error": str(e)}

    def validate_detection_request(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """✅ ACTUALIZADO: Valida parámetros usando configuración centralizada"""
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }

        try:
            # Obtener valores con fallbacks a configuración centralizada
            confidence = request_params.get('confidence_threshold', self.image_config['confidence_threshold'])
            iou = request_params.get('iou_threshold', self.image_config['iou_threshold'])
            max_det = request_params.get('max_detections', self.image_config['max_detections'])

            # Usar rangos de validación centralizados
            conf_range = self.validation_config['confidence_range']
            iou_range = self.validation_config['iou_range']
            det_range = self.validation_config['max_detections_range']
            warnings = self.validation_config['warnings']
            recommendations = self.validation_config['recommendations']

            # Validar umbrales usando configuración centralizada
            if confidence < conf_range[0] or confidence > conf_range[1]:
                validation["errors"].append(
                    f"confidence_threshold debe estar entre {conf_range[0]} y {conf_range[1]}"
                )

            if iou < iou_range[0] or iou > iou_range[1]:
                validation["errors"].append(
                    f"iou_threshold debe estar entre {iou_range[0]} y {iou_range[1]}"
                )

            if max_det < det_range[0] or max_det > det_range[1]:
                validation["errors"].append(
                    f"max_detections debe estar entre {det_range[0]} y {det_range[1]}"
                )

            # Advertencias usando configuración centralizada
            if confidence < warnings['low_confidence']:
                validation["warnings"].append(
                    f"confidence_threshold muy bajo ({confidence}), puede generar muchos falsos positivos"
                )

            if confidence > warnings['high_confidence']:
                validation["warnings"].append(
                    f"confidence_threshold muy alto ({confidence}), puede perderse detecciones válidas"
                )

            # Recomendaciones usando configuración centralizada
            rec_conf_range = recommendations['confidence_range']
            if confidence < rec_conf_range[0] or confidence > rec_conf_range[1]:
                validation["recommendations"].append(
                    f"Para mejores resultados, use confidence_threshold entre {rec_conf_range[0]} y {rec_conf_range[1]}"
                )

            validation["is_valid"] = len(validation["errors"]) == 0

            # Agregar información de configuración usada
            validation["configuration_info"] = {
                "validation_source": "centralized_settings",
                "confidence_range_used": conf_range,
                "iou_range_used": iou_range,
                "warning_thresholds": warnings,
                "recommendation_ranges": recommendations
            }

        except Exception as e:
            validation["is_valid"] = False
            validation["errors"].append(f"Error validando request: {str(e)}")

        return validation

    # ✅ NUEVOS MÉTODOS PARA CONFIGURACIÓN DINÁMICA

    def get_current_config(self) -> Dict[str, Any]:
        """Obtiene la configuración actual del servicio"""
        return {
            "image_detection": self.image_config,
            "validation": self.validation_config,
            "plate_detector": settings.get_plate_detector_config(),
            "char_recognizer": settings.get_char_recognizer_config(),
            "roi": settings.get_roi_config()
        }

    def update_config_from_settings(self):
        """Recarga la configuración desde settings (útil si se cambian valores dinámicamente)"""
        self.image_config = settings.get_image_detection_config()
        self.validation_config = settings.get_validation_config()
        logger.info("🔄 Configuración del DetectionService recargada desde settings")

    def get_recommended_params_for_context(self, context: str = "standard") -> Dict[str, Any]:
        """✅ NUEVO: Obtiene parámetros recomendados según el contexto"""

        context_configs = {
            "standard": self.image_config,
            "quick": settings.get_quick_detection_config(),
            "high_precision": {
                **self.image_config,
                "confidence_threshold": settings.high_confidence_warning,
                "max_detections": settings.max_max_detections
            },
            "high_recall": {
                **self.image_config,
                "confidence_threshold": settings.low_confidence_warning,
                "max_detections": settings.max_max_detections
            }
        }

        return context_configs.get(context, self.image_config)


# Instancia global del servicio
detection_service = DetectionService()