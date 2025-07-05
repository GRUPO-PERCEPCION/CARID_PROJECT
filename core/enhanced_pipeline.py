"""
Pipeline mejorado ajustado para modelos que detectan 6 caracteres sin guión
"""
from typing import Dict, Any, Optional, List  # ✅ AGREGADO List
import numpy as np
import cv2  # ✅ AGREGADO para visualización
from loguru import logger
from core.utils import PerformanceTimer  # ✅ USA TU CLASE EXISTENTE
from .plate_filters import PlateValidator
from .roi_processor import ROIProcessor


class EnhancedALPRPipeline:
    """
    Pipeline ALPR ajustado para modelos de 6 caracteres sin guión
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.plate_validator = PlateValidator()
        self.roi_processor = ROIProcessor(roi_percentage=90.0)
        logger.info("🚀 EnhancedALPRPipeline inicializado para modelos de 6 caracteres sin guión")

    def process_with_enhancements(self, image: np.ndarray, use_roi: bool = True,
                                  filter_six_chars: bool = False,  # ✅ CAMBIAR DEFAULT
                                  return_stats: bool = False,
                                  **model_kwargs) -> Dict[str, Any]:
        """Pipeline optimizado para detección múltiple"""

        try:
            # CONFIGURAR PARA MÚLTIPLES DETECCIONES
            enhanced_kwargs = {
                **model_kwargs,
                'conf': model_kwargs.get('conf', 0.25),  # Más permisivo que antes (era 0.5)
                'iou': model_kwargs.get('iou', 0.3),  # Menos agresivo en NMS (era 0.45)
                'max_det': model_kwargs.get('max_det', 25),  # ✅ MÁS DETECCIONES (era 5-10)
                'verbose': model_kwargs.get('verbose', False)
            }

            logger.info(f"🎯 Pipeline múltiple iniciado: max_det={enhanced_kwargs['max_det']}, "
                        f"conf={enhanced_kwargs['conf']}, iou={enhanced_kwargs['iou']}, "
                        f"filter_six_chars={filter_six_chars}")

            # ROI EXPANDIDO
            if use_roi:
                roi_percentage = 0.85  # ✅ EXPANDIR ROI de 60% a 85%
                roi_image, roi_coords = self._apply_central_roi(image, roi_percentage)
                logger.debug(f"🎯 ROI aplicado: {roi_percentage * 100}% de la imagen")
            else:
                roi_image = image
                roi_coords = None
                logger.debug("🎯 Procesando imagen completa sin ROI")

            # DETECCIÓN DE PLACAS con parámetros mejorados
            logger.debug(f"🔍 Iniciando detección de placas con kwargs: {enhanced_kwargs}")
            plate_result = self.model_manager.detect_plates(roi_image, **enhanced_kwargs)

            # LOG DETALLADO de detecciones
            if plate_result.get("success") and plate_result.get("detections"):
                logger.info(f"📊 Detecciones de placas encontradas: {len(plate_result['detections'])}")
            else:
                logger.warning(f"⚠️ Sin detecciones de placas válidas")

            # PROCESAR TODAS LAS DETECCIONES sin filtrar agresivamente
            final_results = []
            processing_stats = {
                "total_plate_detections": 0,
                "successful_recognitions": 0,
                "failed_recognitions": 0,
                "six_char_validations": 0,
                "auto_formatted": 0
            }

            if plate_result.get("success") and plate_result.get("detections"):
                processing_stats["total_plate_detections"] = len(plate_result["detections"])

                for i, detection in enumerate(plate_result["detections"]):
                    try:
                        logger.debug(f"🔤 Procesando placa {i + 1}/{len(plate_result['detections'])}")

                        # PROCESAR REGIÓN DE PLACA con confianza más baja
                        char_result = self._process_plate_region(roi_image, detection, i)

                        if char_result.get("plate_text") or char_result.get("raw_plate_text"):
                            # VALIDACIÓN MÁS PERMISIVA
                            raw_text = char_result.get("raw_plate_text", "")
                            formatted_text = char_result.get("plate_text", "")

                            # Aceptar si tiene al menos 4 caracteres reconocibles
                            if len(raw_text) >= 4 or len(formatted_text) >= 4:
                                final_results.append(char_result)
                                processing_stats["successful_recognitions"] += 1

                                if char_result.get("six_char_validated", False):
                                    processing_stats["six_char_validations"] += 1

                                if char_result.get("auto_formatted", False):
                                    processing_stats["auto_formatted"] += 1

                                logger.debug(f"✅ Placa {i + 1} aceptada: '{raw_text}' -> '{formatted_text}' "
                                             f"(6chars: {char_result.get('six_char_validated', False)}, "
                                             f"auto: {char_result.get('auto_formatted', False)})")
                            else:
                                processing_stats["failed_recognitions"] += 1
                                logger.debug(f"❌ Placa {i + 1} rechazada: texto muy corto")
                        else:
                            processing_stats["failed_recognitions"] += 1
                            logger.debug(f"❌ Placa {i + 1} sin texto reconocible")

                    except Exception as e:
                        processing_stats["failed_recognitions"] += 1
                        logger.warning(f"⚠️ Error procesando placa {i + 1}: {str(e)}")
                        continue

            # APLICAR FILTROS SEGÚN CONFIGURACIÓN
            if filter_six_chars:
                # Solo aplicar filtro estricto si se solicita explícitamente
                validated_results = [r for r in final_results if r.get("six_char_validated", False)]
                logger.info(f"🔍 Filtro 6 chars aplicado: {len(validated_results)}/{len(final_results)} placas")
            else:
                # Filtro permisivo: aceptar cualquier placa con texto reconocible
                validated_results = [r for r in final_results if len(r.get("raw_plate_text", "")) >= 4]
                logger.info(f"🔍 Filtro permisivo aplicado: {len(validated_results)}/{len(final_results)} placas")

            # ESTADÍSTICAS FINALES
            success = len(validated_results) > 0

            logger.success(f"🎯 Pipeline múltiple completado: {len(validated_results)} placas válidas "
                           f"de {processing_stats['total_plate_detections']} detectadas "
                           f"(ROI: {use_roi}, Filtro: {filter_six_chars})")

            # RESULTADO FINAL COMPLETO
            result = {
                "success": success,
                "total_detections": processing_stats["total_plate_detections"],
                "valid_detections": len(validated_results),
                "final_results": validated_results,  # TODAS las válidas
                "processing_stats": processing_stats,
                "enhancement_info": {
                    "roi_used": use_roi,
                    "roi_percentage": 85.0 if use_roi else 100.0,
                    "filter_applied": filter_six_chars,
                    "detection_mode": "multiple_plates_optimized",
                    "pipeline_version": "enhanced_v2.0"
                },
                "model_params": enhanced_kwargs,
                "validation_summary": {
                    "total_processed": len(final_results),
                    "final_accepted": len(validated_results),
                    "six_char_valid": processing_stats["six_char_validations"],
                    "auto_formatted": processing_stats["auto_formatted"],
                    "success_rate": round(
                        len(validated_results) / max(processing_stats["total_plate_detections"], 1) * 100, 1)
                }
            }

            if return_stats:
                result["detailed_stats"] = {
                    "plate_detection": plate_result,
                    "roi_coords": roi_coords,
                    "processing_time_ms": 0  # Se puede agregar timing si es necesario
                }

            return result

        except Exception as e:
            logger.error(f"❌ Error crítico en pipeline múltiple: {str(e)}")
            logger.exception("Stack trace completo:")
            return {
                "success": False,
                "error": str(e),
                "final_results": [],
                "total_detections": 0,
                "valid_detections": 0,
                "enhancement_info": {
                    "error_occurred": True,
                    "detection_mode": "multiple_plates_failed"
                }
            }

    def _create_empty_result(self, use_roi: bool, roi_coords: Optional[Dict], roi_stats: Optional[Dict],
                             filter_six_chars: bool) -> Dict[str, Any]:
        """Crea resultado vacío con metadatos"""
        return {
            "success": False,
            "message": "No se detectaron placas en la imagen",
            "use_roi": use_roi,
            "roi_coords": roi_coords,
            "filter_six_chars": filter_six_chars,
            "plate_detection": None,
            "final_results": [],
            "model_info": {
                "expects_six_chars": True,
                "detects_dash": False,
                "auto_formatting": filter_six_chars
            },
            "detailed_stats": {
                "roi_stats": roi_stats,
                "validation_detailed": None,
                "processing_method": "roi" if use_roi else "full_image"
            } if roi_stats else None
        }

    def _calculate_combined_confidence(self, plate_conf: float, char_conf: float) -> float:
        """Calcula confianza combinada"""
        return (plate_conf * 0.4) + (char_conf * 0.6)

    def create_visualization(
            self,
            image_input,
            result: Dict[str, Any],
            show_roi: bool = True
    ) -> np.ndarray:
        """
        Crea visualización completa con información de formateo automático
        """
        try:
            # Preprocesar imagen
            image = self.model_manager.plate_detector.preprocess_image(image_input)

            # Crear visualización base
            if result.get("final_results"):
                # Usar visualización existente del detector
                viz_image = self.model_manager.plate_detector.visualize_detections(image)
            else:
                viz_image = image.copy()

            # Agregar visualización del ROI si se usó y se solicita
            if show_roi and result.get("use_roi") and result.get("roi_coords"):
                viz_image = self.roi_processor.visualize_roi(
                    viz_image,
                    result.get("final_results", [])
                )

            return viz_image

        except Exception as e:
            logger.error(f"❌ Error creando visualización: {str(e)}")
            return self.model_manager.plate_detector.preprocess_image(image_input)