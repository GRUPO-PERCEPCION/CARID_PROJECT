"""
Pipeline mejorado COMPLETAMENTE CORREGIDO para modelos que detectan 6 caracteres sin guión
"""
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
from loguru import logger
from core.utils import PerformanceTimer
from .plate_filters import PlateValidator
from .roi_processor import ROIProcessor


class EnhancedALPRPipeline:
    """
    Pipeline ALPR COMPLETAMENTE CORREGIDO para modelos de 6 caracteres sin guión
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.plate_validator = PlateValidator()
        self.roi_processor = ROIProcessor(roi_percentage=85.0)
        logger.info("🚀 EnhancedALPRPipeline COMPLETAMENTE CORREGIDO inicializado")

    def process_with_enhancements(self, image: np.ndarray, use_roi: bool = True,
                                  filter_six_chars: bool = False,
                                  return_stats: bool = False,
                                  **model_kwargs) -> Dict[str, Any]:
        """✅ COMPLETAMENTE CORREGIDO: Pipeline optimizado para detección múltiple"""

        try:
            # CONFIGURAR PARA MÚLTIPLES DETECCIONES MÁS PERMISIVAS
            enhanced_kwargs = {
                **model_kwargs,
                'conf': model_kwargs.get('conf', 0.15),  # ✅ AÚN MÁS PERMISIVO (era 0.25)
                'iou': model_kwargs.get('iou', 0.25),    # ✅ MENOS AGRESIVO en NMS
                'max_det': model_kwargs.get('max_det', 30),  # ✅ MÁS DETECCIONES
                'verbose': model_kwargs.get('verbose', False)
            }

            logger.info(f"🎯 Pipeline CORREGIDO iniciado: max_det={enhanced_kwargs['max_det']}, "
                        f"conf={enhanced_kwargs['conf']}, iou={enhanced_kwargs['iou']}, "
                        f"filter_six_chars={filter_six_chars}")

            # ✅ ROI PROCESSING CORREGIDO
            if use_roi:
                roi_image, roi_coords = self.roi_processor.extract_roi(image)
                logger.debug(f"🎯 ROI aplicado: {self.roi_processor.roi_percentage}% de la imagen")
            else:
                roi_image = image
                roi_coords = None
                logger.debug("🎯 Procesando imagen completa sin ROI")

            # ✅ PASO 1: DETECCIÓN DE PLACAS usando el método correcto
            logger.debug(f"🔍 Iniciando detección de placas con kwargs: {enhanced_kwargs}")

            # ✅ USAR EL MÉTODO CORRECTO del model_manager
            plate_result = self.model_manager.plate_detector.detect_plates(roi_image, **enhanced_kwargs)

            if plate_result.get("success") and plate_result.get("plates_detected", 0) > 0:
                logger.info(f"📊 Detecciones de placas encontradas: {plate_result['plates_detected']}")
            else:
                logger.warning(f"⚠️ Sin detecciones de placas válidas en detector")

            # ✅ PROCESAR CADA PLACA DETECTADA
            final_results = []
            processing_stats = {
                "total_plate_detections": plate_result.get("plates_detected", 0),
                "successful_recognitions": 0,
                "failed_recognitions": 0,
                "six_char_validations": 0,
                "auto_formatted": 0
            }

            if plate_result.get("success") and plate_result.get("plates", []):
                for i, plate_detection in enumerate(plate_result["plates"]):
                    try:
                        logger.debug(f"🔤 Procesando placa {i + 1}/{len(plate_result['plates'])}")

                        # ✅ PASO 2: RECONOCIMIENTO DE CARACTERES usando métodos correctos
                        char_result = self._process_plate_region_corrected(
                            roi_image, plate_detection, i, enhanced_kwargs
                        )

                        if char_result.get("plate_text") or char_result.get("raw_plate_text"):
                            # ✅ COMBINAR RESULTADOS CORRECTAMENTE
                            combined_result = self._combine_detection_and_recognition(
                                plate_detection, char_result, roi_coords
                            )

                            final_results.append(combined_result)
                            processing_stats["successful_recognitions"] += 1

                            # Contar estadísticas especiales
                            if combined_result.get("six_char_validated", False):
                                processing_stats["six_char_validations"] += 1
                            if combined_result.get("auto_formatted", False):
                                processing_stats["auto_formatted"] += 1

                            logger.debug(f"✅ Placa {i + 1} procesada exitosamente: "
                                        f"'{char_result.get('raw_plate_text', '')}' -> "
                                        f"'{char_result.get('plate_text', '')}'")
                        else:
                            processing_stats["failed_recognitions"] += 1
                            logger.debug(f"❌ Placa {i + 1} sin texto reconocible")

                    except Exception as e:
                        processing_stats["failed_recognitions"] += 1
                        logger.warning(f"⚠️ Error procesando placa {i + 1}: {str(e)}")
                        continue

            # ✅ APLICAR FILTROS SEGÚN CONFIGURACIÓN
            if filter_six_chars:
                # Filtro estricto: solo placas de 6 caracteres válidas
                validated_results = [r for r in final_results if r.get("six_char_validated", False)]
                logger.info(f"🔍 Filtro 6 chars aplicado: {len(validated_results)}/{len(final_results)} placas")
            else:
                # Filtro permisivo: cualquier placa con texto reconocible
                validated_results = [r for r in final_results if len(r.get("raw_plate_text", "")) >= 3]
                logger.info(f"🔍 Filtro permisivo aplicado: {len(validated_results)}/{len(final_results)} placas")

            success = len(validated_results) > 0

            logger.success(f"🎯 Pipeline CORREGIDO completado: {len(validated_results)} placas válidas "
                           f"de {processing_stats['total_plate_detections']} detectadas "
                           f"(ROI: {use_roi}, Filtro: {filter_six_chars})")

            # ✅ RESULTADO FINAL COMPLETO
            result = {
                "success": success,
                "total_detections": processing_stats["total_plate_detections"],
                "valid_detections": len(validated_results),
                "final_results": validated_results,
                "processing_stats": processing_stats,
                "enhancement_info": {
                    "roi_used": use_roi,
                    "roi_percentage": self.roi_processor.roi_percentage if use_roi else 100.0,
                    "filter_applied": filter_six_chars,
                    "detection_mode": "multiple_plates_corrected",
                    "pipeline_version": "enhanced_v2.1_corrected"
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
                }

            return result

        except Exception as e:
            logger.error(f"❌ Error crítico en pipeline CORREGIDO: {str(e)}")
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

    def _process_plate_region_corrected(self, roi_image: np.ndarray, plate_detection: Dict[str, Any],
                                       detection_index: int, enhanced_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """✅ NUEVO MÉTODO CORREGIDO: Procesa región de placa usando métodos correctos del model_manager"""
        try:
            # ✅ EXTRAER REGIÓN DE PLACA usando el método correcto
            plate_bbox = plate_detection.get("bbox", [])
            if len(plate_bbox) != 4:
                logger.warning(f"⚠️ Bbox inválido para placa {detection_index}")
                return {"plate_text": "", "raw_plate_text": ""}

            # ✅ USAR EL MÉTODO CORRECTO del plate_detector
            plate_crop = self.model_manager.plate_detector.crop_image_from_bbox(
                roi_image, plate_bbox, padding=10
            )

            if plate_crop.size == 0:
                logger.warning(f"⚠️ Recorte vacío para placa {detection_index}")
                return {"plate_text": "", "raw_plate_text": ""}

            # ✅ RECONOCER CARACTERES usando el método correcto
            # Extraer solo parámetros relevantes para reconocimiento
            char_kwargs = {
                'conf': enhanced_kwargs.get('conf', 0.4),  # Usar confianza más alta para chars
                'verbose': enhanced_kwargs.get('verbose', False)
            }

            char_result = self.model_manager.char_recognizer.recognize_characters(
                plate_crop, **char_kwargs
            )

            # ✅ VALIDAR Y FORMATEAR usando el PlateValidator
            if char_result.get("success") and char_result.get("plate_text"):
                raw_text = char_result["plate_text"]

                # Aplicar validación de 6 caracteres
                validation_result = self.plate_validator.validate_six_characters_only(raw_text)

                if validation_result["is_valid"]:
                    # Texto válido de 6 caracteres
                    return {
                        "plate_text": validation_result["formatted_text"],  # Con guión
                        "raw_plate_text": validation_result["clean_text"],   # Sin guión
                        "confidence": char_result.get("confidence", 0.0),
                        "is_valid_plate": True,
                        "six_char_validated": True,
                        "auto_formatted": True,
                        "validation_info": validation_result,
                        "char_count": validation_result["char_count"]
                    }
                else:
                    # Texto inválido pero mantener para debugging
                    return {
                        "plate_text": raw_text,
                        "raw_plate_text": raw_text,
                        "confidence": char_result.get("confidence", 0.0),
                        "is_valid_plate": False,
                        "six_char_validated": False,
                        "auto_formatted": False,
                        "validation_info": validation_result,
                        "char_count": len(raw_text)
                    }
            else:
                # Sin reconocimiento exitoso
                return {
                    "plate_text": "",
                    "raw_plate_text": "",
                    "confidence": 0.0,
                    "is_valid_plate": False,
                    "six_char_validated": False,
                    "auto_formatted": False,
                    "char_count": 0
                }

        except Exception as e:
            logger.error(f"❌ Error procesando región de placa {detection_index}: {str(e)}")
            return {
                "plate_text": "",
                "raw_plate_text": "",
                "confidence": 0.0,
                "is_valid_plate": False,
                "six_char_validated": False,
                "auto_formatted": False,
                "error": str(e),
                "char_count": 0
            }

    def _combine_detection_and_recognition(self, plate_detection: Dict[str, Any],
                                         char_result: Dict[str, Any],
                                         roi_coords: Optional[Dict[str, int]]) -> Dict[str, Any]:
        """✅ NUEVO MÉTODO: Combina resultados de detección y reconocimiento correctamente"""

        # ✅ AJUSTAR COORDENADAS SI SE USÓ ROI
        final_bbox = plate_detection["bbox"]
        if roi_coords:
            # Ajustar coordenadas del ROI a imagen completa
            final_bbox = [
                final_bbox[0] + roi_coords["x_start"],
                final_bbox[1] + roi_coords["y_start"],
                final_bbox[2] + roi_coords["x_start"],
                final_bbox[3] + roi_coords["y_start"]
            ]

        # ✅ CALCULAR CONFIANZA COMBINADA
        plate_confidence = plate_detection.get("confidence", 0.0)
        char_confidence = char_result.get("confidence", 0.0)
        overall_confidence = self._calculate_combined_confidence(plate_confidence, char_confidence)

        # ✅ RESULTADO COMBINADO COMPLETO
        combined_result = {
            # Información de placa
            "plate_bbox": final_bbox,
            "plate_confidence": plate_confidence,
            "plate_area": plate_detection.get("area", 0),

            # Información de caracteres
            "plate_text": char_result.get("plate_text", ""),           # Con guión (formateado)
            "raw_plate_text": char_result.get("raw_plate_text", ""),   # Sin guión (crudo)
            "character_recognition": {
                "confidence": char_confidence,
                "characters_detected": char_result.get("characters_detected", 0),
                "is_valid_format": char_result.get("is_valid_plate", False)
            },

            # Confianza general
            "overall_confidence": overall_confidence,

            # Validaciones
            "is_valid_plate": char_result.get("is_valid_plate", False),
            "six_char_validated": char_result.get("six_char_validated", False),
            "auto_formatted": char_result.get("auto_formatted", False),

            # Metadatos
            "char_count": char_result.get("char_count", 0),
            "validation_info": char_result.get("validation_info", {}),
            "processing_method": "roi_enhanced_corrected",

            # Info de ROI
            "roi_adjusted": roi_coords is not None,
            "original_roi_bbox": plate_detection["bbox"] if roi_coords else None
        }

        return combined_result

    def _calculate_combined_confidence(self, plate_conf: float, char_conf: float) -> float:
        """Calcula confianza combinada (40% detección, 60% reconocimiento)"""
        return (plate_conf * 0.4) + (char_conf * 0.6)

    def create_visualization(self, image_input, result: Dict[str, Any], show_roi: bool = True) -> np.ndarray:
        """✅ CORREGIDO: Crea visualización completa"""
        try:
            # Preprocesar imagen
            image = self.model_manager.plate_detector.preprocess_image(image_input)

            # Crear copia para visualización
            viz_image = image.copy()

            # ✅ DIBUJAR ROI si se usó
            if show_roi and result.get("enhancement_info", {}).get("roi_used"):
                viz_image = self._draw_roi_overlay(viz_image)

            # ✅ DIBUJAR DETECCIONES
            final_results = result.get("final_results", [])
            if final_results:
                viz_image = self._draw_detections_on_image(viz_image, final_results)

            return viz_image

        except Exception as e:
            logger.error(f"❌ Error creando visualización: {str(e)}")
            return self.model_manager.plate_detector.preprocess_image(image_input)

    def _draw_roi_overlay(self, image: np.ndarray) -> np.ndarray:
        """✅ NUEVO: Dibuja overlay del ROI"""
        try:
            height, width = image.shape[:2]
            roi_percentage = self.roi_processor.roi_percentage / 100.0

            # Calcular dimensiones del ROI
            roi_width = int(width * roi_percentage)
            roi_height = int(height * roi_percentage)

            x_start = (width - roi_width) // 2
            y_start = (height - roi_height) // 2
            x_end = x_start + roi_width
            y_end = y_start + roi_height

            # Dibujar borde del ROI
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)

            # Etiqueta del ROI
            cv2.putText(image, f"ROI {self.roi_processor.roi_percentage}%",
                       (x_start + 5, y_start + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            return image
        except Exception as e:
            logger.debug(f"Error dibujando ROI: {e}")
            return image

    def _draw_detections_on_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """✅ NUEVO: Dibuja detecciones en la imagen"""
        try:
            for i, detection in enumerate(detections):
                bbox = detection.get("plate_bbox", [])
                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = map(int, bbox)

                # Color según validación
                is_valid = detection.get("is_valid_plate", False)
                is_six_char = detection.get("six_char_validated", False)
                auto_formatted = detection.get("auto_formatted", False)

                if is_six_char and auto_formatted:
                    color = (0, 255, 0)  # Verde para 6 chars auto-formateadas
                elif is_valid:
                    color = (0, 255, 255)  # Amarillo para válidas
                else:
                    color = (255, 165, 0)  # Naranja para otras

                # Dibujar rectángulo
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Preparar texto
                plate_text = detection.get("plate_text", "")
                raw_text = detection.get("raw_plate_text", "")
                confidence = detection.get("overall_confidence", 0.0)

                if raw_text and plate_text and raw_text != plate_text:
                    label = f"{raw_text}→{plate_text} ({confidence:.2f})"
                else:
                    label = f"{plate_text or raw_text} ({confidence:.2f})"

                # Indicadores
                indicators = []
                if is_six_char:
                    indicators.append("6C")
                if auto_formatted:
                    indicators.append("AF")
                if indicators:
                    label += f" [{'/'.join(indicators)}]"

                # Dibujar etiqueta con fondo
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0] + 10, y1), color, -1)

                cv2.putText(image, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return image
        except Exception as e:
            logger.debug(f"Error dibujando detecciones: {e}")
            return image