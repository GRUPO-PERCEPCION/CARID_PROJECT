from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from loguru import logger

from .base_model import BaseModel
from config.settings import settings


class PlateDetector(BaseModel):
    """Detector de placas vehiculares usando YOLOv8 con configuración centralizada"""

    def __init__(self, model_path: Optional[str] = None):
        model_path = model_path or settings.plate_model_path
        super().__init__(model_path, "Detector de Placas")

        # ✅ CONFIGURACIONES ESPECÍFICAS DESDE SETTINGS CENTRALIZADOS
        plate_config = settings.get_plate_detector_config()

        self.min_plate_area = plate_config['min_plate_area']
        self.max_detections = plate_config['max_detections']
        self.min_aspect_ratio = plate_config['min_aspect_ratio']
        self.max_aspect_ratio = plate_config['max_aspect_ratio']
        self.confidence_threshold = plate_config['confidence_threshold']
        self.iou_threshold = plate_config['iou_threshold']

        logger.info("🎯 PlateDetector inicializado con configuración centralizada")
        logger.debug(f"📊 Config aplicada: min_area={self.min_plate_area}, "
                     f"max_det={self.max_detections}, aspect_ratio={self.min_aspect_ratio}-{self.max_aspect_ratio}")

    def process_results(self, results: List, original_image: np.ndarray) -> Dict[str, Any]:
        """✅ ACTUALIZADO: Procesa los resultados usando configuración centralizada"""
        try:
            # Extraer bounding boxes
            detections = self.extract_bboxes(results)

            # Filtrar detecciones válidas usando config centralizada
            valid_plates = self._filter_valid_plates(detections, original_image.shape)

            # Preparar resultado final
            result = {
                "success": len(valid_plates) > 0,
                "plates_detected": len(valid_plates),
                "plates": valid_plates,
                "image_shape": {
                    "height": original_image.shape[0],
                    "width": original_image.shape[1],
                    "channels": original_image.shape[2] if len(original_image.shape) > 2 else 1
                },
                "processing_info": {
                    "total_detections": len(detections),
                    "valid_detections": len(valid_plates),
                    "min_area_threshold": self.min_plate_area,
                    "aspect_ratio_range": [self.min_aspect_ratio, self.max_aspect_ratio],
                    "max_detections_limit": self.max_detections,
                    "configuration_source": "centralized_settings"  # ✅ NUEVO
                }
            }

            logger.info(f"🎯 Placas detectadas: {len(valid_plates)}/{len(detections)} "
                        f"(usando config centralizada)")
            return result

        except Exception as e:
            logger.error(f"❌ Error procesando resultados de placas: {str(e)}")
            return {
                "success": False,
                "plates_detected": 0,
                "plates": [],
                "error": str(e),
                "configuration_source": "centralized_settings"
            }

    def _filter_valid_plates(self, detections: List[Dict], image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """✅ ACTUALIZADO: Filtra las detecciones usando configuración centralizada"""
        valid_plates = []

        for detection in detections:
            # ✅ USAR MIN_PLATE_AREA CENTRALIZADO
            if detection["area"] < self.min_plate_area:
                logger.debug(f"🔍 Placa descartada por área pequeña: {detection['area']} < {self.min_plate_area}")
                continue

            # Verificar aspect ratio usando config centralizada
            bbox = detection["bbox"]
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])

            if height == 0:  # Evitar división por cero
                continue

            aspect_ratio = width / height

            # ✅ USAR ASPECT_RATIO CENTRALIZADO
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                logger.debug(f"🔍 Placa descartada por aspect ratio: {aspect_ratio:.2f} "
                             f"(rango válido: {self.min_aspect_ratio}-{self.max_aspect_ratio})")
                continue

            # Verificar que el bbox esté dentro de la imagen
            if not self._is_bbox_valid(bbox, image_shape):
                logger.debug(f"🔍 Placa descartada por bbox inválido: {bbox}")
                continue

            # Agregar información adicional
            enhanced_detection = {
                **detection,
                "aspect_ratio": aspect_ratio,
                "width": width,
                "height": height,
                "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                "normalized_bbox": self._normalize_bbox(bbox, image_shape),
                "validation_passed": {  # ✅ NUEVA INFO DE VALIDACIÓN
                    "min_area": detection["area"] >= self.min_plate_area,
                    "aspect_ratio": self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio,
                    "bbox_valid": True,
                    "config_source": "centralized_settings"
                }
            }

            valid_plates.append(enhanced_detection)

        # ✅ ORDENAR Y LIMITAR USANDO CONFIG CENTRALIZADA
        valid_plates.sort(key=lambda x: x["confidence"], reverse=True)
        limited_plates = valid_plates[:self.max_detections]

        if len(valid_plates) > self.max_detections:
            logger.info(f"🔍 Limitando detecciones: {len(valid_plates)} -> {self.max_detections} "
                        f"(según config centralizada)")

        return limited_plates

    def _is_bbox_valid(self, bbox: List[float], image_shape: Tuple[int, int, int]) -> bool:
        """Verifica que el bounding box esté dentro de los límites de la imagen"""
        x1, y1, x2, y2 = bbox
        height, width = image_shape[:2]

        return (0 <= x1 < width and 0 <= y1 < height and
                0 <= x2 < width and 0 <= y2 < height and
                x1 < x2 and y1 < y2)

    def _normalize_bbox(self, bbox: List[float], image_shape: Tuple[int, int, int]) -> List[float]:
        """Normaliza las coordenadas del bbox a valores entre 0 y 1"""
        x1, y1, x2, y2 = bbox
        height, width = image_shape[:2]

        return [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height
        ]

    def detect_plates(self, image_input, **kwargs) -> Dict[str, Any]:
        """
        ✅ ACTUALIZADO: Detecta placas usando configuración centralizada con fallbacks
        """
        try:
            # Preprocesar imagen
            image = self.preprocess_image(image_input)

            # ✅ APLICAR CONFIGURACIÓN CENTRALIZADA CON FALLBACKS
            model_kwargs = self._build_detection_kwargs(kwargs)

            # Realizar predicción
            logger.info(f"🔍 Detectando placas con config centralizada: {model_kwargs}")
            results = self.predict(image, **model_kwargs)

            # Procesar resultados
            processed_results = self.process_results(results, image)

            return processed_results

        except Exception as e:
            logger.error(f"❌ Error en detección de placas: {str(e)}")
            return {
                "success": False,
                "plates_detected": 0,
                "plates": [],
                "error": str(e),
                "configuration_source": "centralized_settings"
            }

    def _build_detection_kwargs(self, user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """✅ NUEVO: Construye kwargs usando configuración centralizada como fallback"""

        # Usar config centralizada como base
        detection_kwargs = {
            'conf': user_kwargs.get('conf', self.confidence_threshold),
            'iou': user_kwargs.get('iou', self.iou_threshold),
            'verbose': user_kwargs.get('verbose', False)
        }

        logger.debug(f"🔧 Config base PlateDetector: conf={self.confidence_threshold}, iou={self.iou_threshold}")
        logger.debug(f"📝 User kwargs: {user_kwargs}")
        logger.debug(f"⚙️ Detection kwargs finales: {detection_kwargs}")

        return detection_kwargs

    def extract_plate_regions(self, image_input, **kwargs) -> List[Tuple[np.ndarray, Dict]]:
        """
        ✅ ACTUALIZADO: Detecta placas y extrae las regiones recortadas usando config centralizada
        """
        try:
            # Detectar placas
            detection_result = self.detect_plates(image_input, **kwargs)

            if not detection_result["success"]:
                return []

            # Cargar imagen original
            image = self.preprocess_image(image_input)

            # Extraer regiones
            plate_regions = []
            for plate_info in detection_result["plates"]:
                try:
                    # Recortar región de la placa
                    cropped_plate = self.crop_image_from_bbox(
                        image,
                        plate_info["bbox"],
                        padding=10
                    )

                    plate_regions.append((cropped_plate, plate_info))

                except Exception as e:
                    logger.warning(f"⚠️ Error extrayendo región de placa: {str(e)}")
                    continue

            logger.info(f"✂️ Extraídas {len(plate_regions)} regiones de placas (config centralizada)")
            return plate_regions

        except Exception as e:
            logger.error(f"❌ Error extrayendo regiones de placas: {str(e)}")
            return []

    def get_best_plate(self, image_input, **kwargs) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        ✅ ACTUALIZADO: Detecta y retorna la mejor placa usando configuración centralizada
        """
        plate_regions = self.extract_plate_regions(image_input, **kwargs)

        if not plate_regions:
            return None

        # Retornar la primera (mayor confianza según config centralizada)
        best_plate = plate_regions[0]
        logger.info(f"🏆 Mejor placa seleccionada con confianza: {best_plate[1]['confidence']:.3f} "
                    f"(usando config centralizada)")

        return best_plate

    def visualize_detections(self, image_input, **kwargs) -> np.ndarray:
        """
        ✅ ACTUALIZADO: Crea una imagen con las detecciones visualizadas usando config centralizada
        """
        try:
            # Detectar placas
            detection_result = self.detect_plates(image_input, **kwargs)

            # Cargar imagen original
            image = self.preprocess_image(image_input)
            result_image = image.copy()

            if not detection_result["success"]:
                return result_image

            # ✅ AGREGAR INFORMACIÓN DE CONFIG EN LA VISUALIZACIÓN
            self._add_config_info_to_image(result_image, detection_result)

            # Dibujar detecciones
            for i, plate in enumerate(detection_result["plates"]):
                bbox = plate["bbox"]
                confidence = plate["confidence"]
                aspect_ratio = plate.get("aspect_ratio", 0)

                # Coordenadas del rectángulo
                x1, y1, x2, y2 = map(int, bbox)

                # ✅ COLOR BASADO EN VALIDACIÓN CENTRALIZADA
                validation = plate.get("validation_passed", {})
                if all(validation.values()):
                    color_intensity = int(255 * confidence)
                    color = (0, color_intensity, 0)  # Verde para placas completamente válidas
                else:
                    color = (255, 165, 0)  # Naranja para placas parcialmente válidas

                # Dibujar rectángulo
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

                # ✅ TEXTO CON INFORMACIÓN DE CONFIG
                label = f"Placa {i + 1}: {confidence:.2f}"
                if aspect_ratio > 0:
                    label += f" (AR:{aspect_ratio:.1f})"

                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                # Fondo para el texto
                cv2.rectangle(
                    result_image,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )

                # Texto
                cv2.putText(
                    result_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # Blanco
                    2
                )

                # ✅ AGREGAR INDICADORES DE VALIDACIÓN
                self._add_validation_indicators(result_image, plate, x1, y2)

            return result_image

        except Exception as e:
            logger.error(f"❌ Error visualizando detecciones: {str(e)}")
            return self.preprocess_image(image_input)

    def _add_config_info_to_image(self, image: np.ndarray, detection_result: Dict[str, Any]):
        """✅ NUEVO: Agrega información de configuración a la imagen"""
        try:
            config_text = f"Config: area>={self.min_plate_area}, AR={self.min_aspect_ratio}-{self.max_aspect_ratio}"

            # Fondo para el texto de configuración
            text_size = cv2.getTextSize(config_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (10, 10), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)

            # Texto de configuración
            cv2.putText(
                image,
                config_text,
                (15, text_size[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            # Información de detecciones
            det_info = f"Detectadas: {detection_result['plates_detected']}/{detection_result['processing_info']['total_detections']}"
            cv2.putText(
                image,
                det_info,
                (15, text_size[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        except Exception as e:
            logger.debug(f"Error agregando info de config: {e}")

    def _add_validation_indicators(self, image: np.ndarray, plate: Dict[str, Any], x: int, y: int):
        """✅ NUEVO: Agrega indicadores visuales de validación"""
        try:
            validation = plate.get("validation_passed", {})

            # Indicadores de validación
            indicators = []
            if validation.get("min_area", False):
                indicators.append("A")  # Area
            if validation.get("aspect_ratio", False):
                indicators.append("R")  # Ratio
            if validation.get("bbox_valid", False):
                indicators.append("B")  # Bbox

            if indicators:
                indicator_text = "✓" + "".join(indicators)
                cv2.putText(
                    image,
                    indicator_text,
                    (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1
                )
        except Exception as e:
            logger.debug(f"Error agregando indicadores: {e}")

    # ✅ NUEVOS MÉTODOS PARA CONFIGURACIÓN DINÁMICA

    def get_current_config(self) -> Dict[str, Any]:
        """✅ NUEVO: Obtiene la configuración actual del detector"""
        return {
            "min_plate_area": self.min_plate_area,
            "max_detections": self.max_detections,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "configuration_source": "centralized_settings"
        }

    def update_config_from_settings(self):
        """✅ NUEVO: Recarga la configuración desde settings"""
        plate_config = settings.get_plate_detector_config()

        self.min_plate_area = plate_config['min_plate_area']
        self.max_detections = plate_config['max_detections']
        self.min_aspect_ratio = plate_config['min_aspect_ratio']
        self.max_aspect_ratio = plate_config['max_aspect_ratio']
        self.confidence_threshold = plate_config['confidence_threshold']
        self.iou_threshold = plate_config['iou_threshold']

        logger.info("🔄 Configuración del PlateDetector recargada desde settings")

    def validate_detection_params(self, **kwargs) -> Dict[str, Any]:
        """✅ NUEVO: Valida parámetros de detección"""
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "config_applied": self.get_current_config()
        }

        try:
            conf = kwargs.get('conf', self.confidence_threshold)
            iou = kwargs.get('iou', self.iou_threshold)

            # Validaciones básicas
            if conf < 0.1 or conf > 1.0:
                validation["errors"].append("confidence debe estar entre 0.1 y 1.0")

            if iou < 0.1 or iou > 1.0:
                validation["errors"].append("iou debe estar entre 0.1 y 1.0")

            # Advertencias
            if conf < 0.3:
                validation["warnings"].append("confidence bajo puede generar muchos falsos positivos")

            if conf > 0.8:
                validation["warnings"].append("confidence alto puede perder detecciones válidas")

            validation["is_valid"] = len(validation["errors"]) == 0

        except Exception as e:
            validation["is_valid"] = False
            validation["errors"].append(f"Error validando parámetros: {str(e)}")

        return validation

    def get_detection_statistics(self, image_input, **kwargs) -> Dict[str, Any]:
        """✅ NUEVO: Obtiene estadísticas detalladas de detección"""
        try:
            detection_result = self.detect_plates(image_input, **kwargs)

            if not detection_result["success"]:
                return {
                    "total_detections": 0,
                    "valid_detections": 0,
                    "configuration_used": self.get_current_config()
                }

            plates = detection_result["plates"]

            # Calcular estadísticas
            areas = [p["area"] for p in plates]
            confidences = [p["confidence"] for p in plates]
            aspect_ratios = [p.get("aspect_ratio", 0) for p in plates]

            return {
                "total_detections": detection_result["processing_info"]["total_detections"],
                "valid_detections": len(plates),
                "statistics": {
                    "confidence": {
                        "min": min(confidences) if confidences else 0,
                        "max": max(confidences) if confidences else 0,
                        "avg": sum(confidences) / len(confidences) if confidences else 0
                    },
                    "areas": {
                        "min": min(areas) if areas else 0,
                        "max": max(areas) if areas else 0,
                        "avg": sum(areas) / len(areas) if areas else 0
                    },
                    "aspect_ratios": {
                        "min": min(aspect_ratios) if aspect_ratios else 0,
                        "max": max(aspect_ratios) if aspect_ratios else 0,
                        "avg": sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 0
                    }
                },
                "configuration_used": self.get_current_config(),
                "validation_summary": {
                    "plates_passed_area_filter": len([p for p in plates if p["area"] >= self.min_plate_area]),
                    "plates_passed_ratio_filter": len([p for p in plates
                                                       if self.min_aspect_ratio <= p.get("aspect_ratio",
                                                                                         0) <= self.max_aspect_ratio]),
                    "plates_within_limit": min(len(plates), self.max_detections)
                }
            }

        except Exception as e:
            logger.error(f"❌ Error calculando estadísticas: {str(e)}")
            return {
                "error": str(e),
                "configuration_used": self.get_current_config()
            }

    def optimize_for_context(self, context: str = "standard"):
        """✅ NUEVO: Optimiza configuración para diferentes contextos"""
        try:
            if context == "high_precision":
                # Configuración para máxima precisión
                self.min_plate_area = int(self.min_plate_area * 1.5)
                self.min_aspect_ratio = max(2.0, self.min_aspect_ratio)
                self.confidence_threshold = min(0.8, self.confidence_threshold + 0.2)

            elif context == "high_recall":
                # Configuración para capturar más placas
                self.min_plate_area = int(self.min_plate_area * 0.7)
                self.min_aspect_ratio = max(1.2, self.min_aspect_ratio - 0.3)
                self.max_aspect_ratio = min(8.0, self.max_aspect_ratio + 1.0)
                self.confidence_threshold = max(0.3, self.confidence_threshold - 0.2)
                self.max_detections = min(10, self.max_detections + 5)

            elif context == "speed":
                # Configuración para velocidad
                self.max_detections = min(3, self.max_detections)
                self.confidence_threshold = min(0.7, self.confidence_threshold + 0.1)

            logger.info(f"🎯 PlateDetector optimizado para contexto '{context}'")

        except Exception as e:
            logger.error(f"❌ Error optimizando para contexto '{context}': {str(e)}")
            # Recargar configuración original si hay error
            self.update_config_from_settings()