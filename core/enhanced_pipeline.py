"""
Pipeline mejorado que integra ROI + Filtros + tu utils.py existente
"""
from typing import Dict, Any, Optional
import numpy as np
from loguru import logger
from core.utils import PerformanceTimer  # ✅ USA TU CLASE EXISTENTE
from .plate_filters import PlateValidator
from .roi_processor import ROIProcessor


class EnhancedALPRPipeline:
    """
    Pipeline ALPR mejorado que integra todas las funcionalidades
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.plate_validator = PlateValidator()
        self.roi_processor = ROIProcessor(roi_percentage=10.0)
        logger.info("🚀 EnhancedALPRPipeline inicializado")

    def process_with_enhancements(
            self,
            image_input,
            use_roi: bool = False,
            filter_six_chars: bool = True,
            return_stats: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Procesa imagen con todas las mejoras integradas

        Args:
            image_input: Imagen de entrada
            use_roi: Si usar ROI central (para video/streaming)
            filter_six_chars: Si filtrar por 6 caracteres exactos
            return_stats: Si incluir estadísticas detalladas
        """

        with PerformanceTimer(f"Pipeline {'con ROI' if use_roi else 'completo'}"):
            try:
                logger.info(f"🔄 Procesando {'con ROI' if use_roi else 'imagen completa'} "
                            f"{'+ filtro 6 chars' if filter_six_chars else ''}")

                # Preprocesar imagen
                image = self.model_manager.plate_detector.preprocess_image(image_input)
                original_image = image.copy()
                processing_image = image.copy()

                # Variables para tracking
                roi_coords = None
                roi_stats = None

                # Aplicar ROI si se solicita
                if use_roi:
                    processing_image, roi_coords = self.roi_processor.extract_roi(image)
                    logger.info(f"🎯 Procesando ROI de {roi_coords['width']}x{roi_coords['height']}")

                    if return_stats:
                        roi_stats = self.roi_processor.get_roi_statistics((image.shape[0], image.shape[1]))

                # Paso 1: Detectar placas en la imagen (o ROI)
                with PerformanceTimer("Detección de placas"):
                    plate_results = self.model_manager.plate_detector.detect_plates(
                        processing_image, **kwargs
                    )

                if not plate_results["success"] or plate_results["plates_detected"] == 0:
                    return self._create_empty_result(use_roi, roi_coords, roi_stats, filter_six_chars)

                # Paso 2: Procesar cada placa detectada
                final_results = []
                validation_stats = {"total_plates": 0, "six_char_valid": 0, "filtered_out": 0}

                with PerformanceTimer("Reconocimiento de caracteres"):
                    for i, plate_info in enumerate(plate_results["plates"]):
                        try:
                            # Extraer región de la placa
                            plate_region = self.model_manager.plate_detector.crop_image_from_bbox(
                                processing_image, plate_info["bbox"], padding=10
                            )

                            # Reconocer caracteres
                            char_results = self.model_manager.char_recognizer.recognize_characters(
                                plate_region, **kwargs
                            )

                            validation_stats["total_plates"] += 1

                            # Aplicar filtro de 6 caracteres si se solicita
                            if filter_six_chars:
                                validation = self.plate_validator.validate_six_characters_only(
                                    char_results.get("plate_text", "")
                                )

                                if not validation["is_valid"]:
                                    validation_stats["filtered_out"] += 1
                                    logger.debug(f"❌ Placa rechazada por filtro: {char_results.get('plate_text', '')} "
                                                 f"- {validation['reason']}")
                                    continue

                                # Actualizar texto limpio
                                char_results["plate_text"] = validation["clean_text"]
                                char_results["validation_info"] = validation
                                validation_stats["six_char_valid"] += 1

                            # Combinar resultados
                            combined_result = {
                                "plate_id": i + 1,
                                "plate_bbox": plate_info["bbox"],
                                "plate_confidence": plate_info["confidence"],
                                "plate_area": plate_info["area"],
                                "character_recognition": char_results,
                                "plate_text": char_results.get("plate_text", ""),
                                "overall_confidence": self._calculate_combined_confidence(
                                    plate_info["confidence"],
                                    char_results.get("confidence", 0.0)
                                ),
                                "is_valid_plate": char_results.get("is_valid_format", False),
                                "six_char_validated": filter_six_chars,
                                "processing_method": "roi" if use_roi else "full_image"
                            }

                            final_results.append(combined_result)

                        except Exception as e:
                            logger.error(f"❌ Error procesando placa {i + 1}: {str(e)}")
                            continue

                # Ajustar coordenadas si se usó ROI
                if use_roi and roi_coords and final_results:
                    with PerformanceTimer("Ajuste de coordenadas ROI"):
                        final_results = self.roi_processor.adjust_detections_to_full_image(
                            final_results, roi_coords
                        )

                # Ordenar por confianza
                final_results.sort(key=lambda x: x["overall_confidence"], reverse=True)

                # Resultado final
                result = {
                    "success": len(final_results) > 0,
                    "plates_processed": len(final_results),
                    "use_roi": use_roi,
                    "roi_coords": roi_coords,
                    "filter_six_chars": filter_six_chars,
                    "plate_detection": plate_results,
                    "final_results": final_results,
                    "best_result": final_results[0] if final_results else None,
                    "processing_summary": {
                        "plates_detected": plate_results["plates_detected"],
                        "plates_with_text": len(final_results),
                        "valid_plates": len([r for r in final_results if r["is_valid_plate"]]),
                        "six_char_filter_applied": filter_six_chars,
                        "validation_stats": validation_stats
                    }
                }

                # Agregar estadísticas si se solicitan
                if return_stats:
                    result["detailed_stats"] = {
                        "roi_stats": roi_stats,
                        "validation_detailed": self.plate_validator.get_validation_stats(
                            final_results) if final_results else None,
                        "processing_method": "roi" if use_roi else "full_image",
                        "enhancement_flags": {
                            "roi_enabled": use_roi,
                            "six_char_filter": filter_six_chars,
                            "stats_requested": return_stats
                        }
                    }

                return result

            except Exception as e:
                logger.error(f"❌ Error en pipeline mejorado: {str(e)}")
                return {
                    "success": False,
                    "message": f"Error en pipeline: {str(e)}",
                    "use_roi": use_roi,
                    "final_results": []
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
        Crea visualización completa usando tu lógica existente + nuevas características
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