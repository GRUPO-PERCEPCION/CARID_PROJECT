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

    def process_with_enhancements(
            self,
            image_input,
            use_roi: bool = False,
            filter_six_chars: bool = True,
            return_stats: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Procesa imagen esperando exactamente 6 caracteres del modelo
        """

        with PerformanceTimer(f"Pipeline {'con ROI' if use_roi else 'completo'} - 6 chars sin guión"):
            try:
                logger.info(f"🔄 Procesando {'con ROI' if use_roi else 'imagen completa'} "
                            f"{'+ filtro 6 chars (sin guión)' if filter_six_chars else ''}")

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
                    logger.info(f"🎯 ROI aplicado: {roi_coords['width']}x{roi_coords['height']} "
                                f"({self.roi_processor.roi_percentage}% del total)")

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
                validation_stats = {"total_plates": 0, "six_char_valid": 0, "filtered_out": 0, "auto_formatted": 0}

                with PerformanceTimer("Reconocimiento de caracteres (6 chars sin guión)"):
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

                            # ✅ NUEVO: Procesar texto crudo (6 caracteres sin guión)
                            raw_plate_text = char_results.get("plate_text", "")

                            logger.debug(f"🔤 Texto crudo del modelo: '{raw_plate_text}'")

                            # Aplicar filtro de 6 caracteres si se solicita
                            if filter_six_chars:
                                validation = self.plate_validator.validate_six_characters_only(raw_plate_text)

                                if not validation["is_valid"]:
                                    validation_stats["filtered_out"] += 1
                                    logger.debug(f"❌ Placa rechazada: '{raw_plate_text}' - {validation['reason']}")
                                    continue

                                # ✅ USAR TEXTO FORMATEADO (con guión agregado automáticamente)
                                formatted_text = validation["formatted_text"]
                                char_results["plate_text"] = formatted_text
                                char_results["raw_plate_text"] = validation["clean_text"]  # Original
                                char_results["validation_info"] = validation
                                char_results["auto_formatted"] = True

                                validation_stats["six_char_valid"] += 1
                                validation_stats["auto_formatted"] += 1

                                logger.info(f"✅ Placa formateada: '{validation['clean_text']}' -> '{formatted_text}'")
                            else:
                                # Sin filtro, usar texto tal como viene
                                char_results["raw_plate_text"] = raw_plate_text
                                char_results["auto_formatted"] = False

                            # Combinar resultados
                            combined_result = {
                                "plate_id": i + 1,
                                "plate_bbox": plate_info["bbox"],
                                "plate_confidence": plate_info["confidence"],
                                "plate_area": plate_info["area"],
                                "character_recognition": char_results,
                                "plate_text": char_results.get("plate_text", ""),
                                "raw_plate_text": char_results.get("raw_plate_text", ""),  # ✅ NUEVO
                                "overall_confidence": self._calculate_combined_confidence(
                                    plate_info["confidence"],
                                    char_results.get("confidence", 0.0)
                                ),
                                "is_valid_plate": char_results.get("is_valid_format", False),
                                "six_char_validated": filter_six_chars and validation_stats["six_char_valid"] > 0,
                                "auto_formatted": char_results.get("auto_formatted", False),  # ✅ NUEVO
                                "validation_info": char_results.get("validation_info", {}),  # ✅ NUEVO
                                "processing_method": "roi" if use_roi else "full_image",
                                "model_output": raw_plate_text  # ✅ GUARDAR OUTPUT ORIGINAL
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
                        "auto_formatted_plates": validation_stats["auto_formatted"],  # ✅ NUEVO
                        "validation_stats": validation_stats
                    },
                    "model_info": {  # ✅ INFORMACIÓN DEL MODELO
                        "expects_six_chars": True,
                        "detects_dash": False,
                        "auto_formatting": filter_six_chars
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
                            "auto_dash_formatting": filter_six_chars,  # ✅ NUEVO
                            "stats_requested": return_stats
                        }
                    }

                # ✅ LOG MEJORADO
                if final_results:
                    best = final_results[0]
                    logger.success(f"✅ Pipeline completado: {len(final_results)} placa(s). "
                                 f"Mejor: '{best.get('raw_plate_text', '')}' -> '{best['plate_text']}' "
                                 f"(Confianza: {best['overall_confidence']:.3f})")
                else:
                    logger.info("📭 No se detectaron placas válidas")

                return result

            except Exception as e:
                logger.error(f"❌ Error en pipeline mejorado: {str(e)}")
                return {
                    "success": False,
                    "message": f"Error en pipeline: {str(e)}",
                    "use_roi": use_roi,
                    "final_results": [],
                    "model_info": {
                        "expects_six_chars": True,
                        "detects_dash": False,
                        "auto_formatting": filter_six_chars
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