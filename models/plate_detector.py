from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from loguru import logger

from .base_model import BaseModel
from config.settings import settings


class PlateDetector(BaseModel):
    """Detector de placas vehiculares usando YOLOv8"""

    def __init__(self, model_path: Optional[str] = None):
        model_path = model_path or settings.plate_model_path
        super().__init__(model_path, "Detector de Placas")

        # Configuraciones específicas para detección de placas
        self.min_plate_area = 500  # Área mínima para considerar una detección válida
        self.max_detections = 5  # Máximo número de placas a detectar por imagen

    def process_results(self, results: List, original_image: np.ndarray) -> Dict[str, Any]:
        """Procesa los resultados de detección de placas"""
        try:
            # Extraer bounding boxes
            detections = self.extract_bboxes(results)

            # Filtrar detecciones válidas
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
                    "min_area_threshold": self.min_plate_area
                }
            }

            logger.info(f"🎯 Placas detectadas: {len(valid_plates)}/{len(detections)}")
            return result

        except Exception as e:
            logger.error(f"❌ Error procesando resultados de placas: {str(e)}")
            return {
                "success": False,
                "plates_detected": 0,
                "plates": [],
                "error": str(e)
            }

    def _filter_valid_plates(self, detections: List[Dict], image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Filtra las detecciones para obtener solo placas válidas"""
        valid_plates = []

        for detection in detections:
            # Verificar área mínima
            if detection["area"] < self.min_plate_area:
                logger.debug(f"🔍 Placa descartada por área pequeña: {detection['area']}")
                continue

            # Verificar aspect ratio (placas son típicamente rectangulares)
            bbox = detection["bbox"]
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])

            if height == 0:  # Evitar división por cero
                continue

            aspect_ratio = width / height

            # Placas peruanas típicamente tienen aspect ratio entre 2:1 y 5:1
            if not (1.5 <= aspect_ratio <= 6.0):
                logger.debug(f"🔍 Placa descartada por aspect ratio: {aspect_ratio:.2f}")
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
                "normalized_bbox": self._normalize_bbox(bbox, image_shape)
            }

            valid_plates.append(enhanced_detection)

        # Ordenar por confianza y limitar número máximo
        valid_plates.sort(key=lambda x: x["confidence"], reverse=True)
        return valid_plates[:self.max_detections]

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
        Detecta placas en una imagen

        Args:
            image_input: Puede ser path de imagen (str) o numpy array
            **kwargs: Parámetros adicionales para la predicción

        Returns:
            Dict con resultados de detección
        """
        try:
            # Preprocesar imagen
            image = self.preprocess_image(image_input)

            # Realizar predicción
            logger.info("🔍 Detectando placas...")
            results = self.predict(image, **kwargs)

            # Procesar resultados
            processed_results = self.process_results(results, image)

            return processed_results

        except Exception as e:
            logger.error(f"❌ Error en detección de placas: {str(e)}")
            return {
                "success": False,
                "plates_detected": 0,
                "plates": [],
                "error": str(e)
            }

    def extract_plate_regions(self, image_input, **kwargs) -> List[Tuple[np.ndarray, Dict]]:
        """
        Detecta placas y extrae las regiones recortadas

        Returns:
            Lista de tuplas (imagen_recortada, info_detección)
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

            logger.info(f"✂️ Extraídas {len(plate_regions)} regiones de placas")
            return plate_regions

        except Exception as e:
            logger.error(f"❌ Error extrayendo regiones de placas: {str(e)}")
            return []

    def get_best_plate(self, image_input, **kwargs) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Detecta y retorna la mejor placa (mayor confianza)

        Returns:
            Tupla (imagen_recortada, info_detección) o None si no se detecta nada
        """
        plate_regions = self.extract_plate_regions(image_input, **kwargs)

        if not plate_regions:
            return None

        # Retornar la primera (mayor confianza)
        best_plate = plate_regions[0]
        logger.info(f"🏆 Mejor placa seleccionada con confianza: {best_plate[1]['confidence']:.3f}")

        return best_plate

    def visualize_detections(self, image_input, **kwargs) -> np.ndarray:
        """
        Crea una imagen con las detecciones visualizadas

        Returns:
            Imagen con bounding boxes dibujados
        """
        try:
            # Detectar placas
            detection_result = self.detect_plates(image_input, **kwargs)

            # Cargar imagen original
            image = self.preprocess_image(image_input)
            result_image = image.copy()

            if not detection_result["success"]:
                return result_image

            # Dibujar detecciones
            for i, plate in enumerate(detection_result["plates"]):
                bbox = plate["bbox"]
                confidence = plate["confidence"]

                # Coordenadas del rectángulo
                x1, y1, x2, y2 = map(int, bbox)

                # Color basado en confianza (verde más intenso = mayor confianza)
                color_intensity = int(255 * confidence)
                color = (0, color_intensity, 0)  # Verde en RGB

                # Dibujar rectángulo
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

                # Texto con información
                label = f"Placa {i + 1}: {confidence:.2f}"
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

            return result_image

        except Exception as e:
            logger.error(f"❌ Error visualizando detecciones: {str(e)}")
            return self.preprocess_image(image_input)