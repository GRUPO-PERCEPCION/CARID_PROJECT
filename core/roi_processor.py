"""
Procesador de ROI que utiliza funciones de tu utils.py
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from loguru import logger
from core.utils import (  # ✅ USA TUS FUNCIONES EXISTENTES
    calculate_bbox_area,
    normalize_bbox_coordinates,
    calculate_bbox_iou,
    filter_overlapping_detections
)


class ROIProcessor:
    """Procesador de ROI que integra con tu sistema existente"""

    def __init__(self, roi_percentage: float = 10.0):
        """
        Args:
            roi_percentage: Porcentaje del centro de la imagen para ROI
        """
        self.roi_percentage = roi_percentage
        logger.info(f"🎯 ROIProcessor inicializado - ROI central: {roi_percentage}%")

    def calculate_central_roi(self, image_shape: Tuple[int, int]) -> Dict[str, int]:
        """
        Calcula las coordenadas del ROI central
        NUEVA FUNCIÓN específica para ROI central
        """
        height, width = image_shape

        # Calcular tamaño del ROI basado en porcentaje
        roi_width = int(width * (self.roi_percentage / 100))
        roi_height = int(height * (self.roi_percentage / 100))

        # Asegurar dimensiones mínimas
        roi_width = max(roi_width, 200)  # Mínimo 200px de ancho
        roi_height = max(roi_height, 150)  # Mínimo 150px de alto

        # Centrar el ROI
        x_start = (width - roi_width) // 2
        y_start = (height - roi_height) // 2
        x_end = x_start + roi_width
        y_end = y_start + roi_height

        # Asegurar que está dentro de la imagen
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(width, x_end)
        y_end = min(height, y_end)

        return {
            "x_start": x_start,
            "y_start": y_start,
            "x_end": x_end,
            "y_end": y_end,
            "width": x_end - x_start,
            "height": y_end - y_start,
            "center_x": width // 2,
            "center_y": height // 2,
            "roi_percentage_used": self.roi_percentage,
            "original_width": width,
            "original_height": height
        }

    def extract_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Extrae la región de interés de la imagen
        NUEVA FUNCIÓN para ROI
        """
        height, width = image.shape[:2]
        roi_coords = self.calculate_central_roi((height, width))

        # Extraer ROI
        roi_image = image[
                    roi_coords["y_start"]:roi_coords["y_end"],
                    roi_coords["x_start"]:roi_coords["x_end"]
                    ]

        logger.debug(f"🎯 ROI extraído: {roi_coords['width']}x{roi_coords['height']} "
                     f"en posición ({roi_coords['x_start']}, {roi_coords['y_start']})")

        return roi_image, roi_coords

    def adjust_detections_to_full_image(
            self,
            detections: List[Dict[str, Any]],
            roi_coords: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """
        Ajusta las coordenadas de detecciones del ROI al tamaño completo
        USA tus funciones existentes para cálculos de bbox
        """
        adjusted_detections = []

        for detection in detections:
            adjusted_detection = detection.copy()

            # Ajustar bbox del ROI a coordenadas de imagen completa
            if "plate_bbox" in detection:
                roi_bbox = detection["plate_bbox"]

                # Coordenadas originales en ROI
                x1_roi, y1_roi, x2_roi, y2_roi = roi_bbox

                # Convertir a coordenadas de imagen completa
                x1_full = x1_roi + roi_coords["x_start"]
                y1_full = y1_roi + roi_coords["y_start"]
                x2_full = x2_roi + roi_coords["x_start"]
                y2_full = y2_roi + roi_coords["y_start"]

                full_bbox = [x1_full, y1_full, x2_full, y2_full]

                adjusted_detection["plate_bbox"] = full_bbox
                adjusted_detection["roi_adjusted"] = True
                adjusted_detection["original_roi_bbox"] = roi_bbox

                # ✅ USA TU FUNCIÓN EXISTENTE para calcular área
                adjusted_detection["plate_area"] = calculate_bbox_area(full_bbox)

                # ✅ USA TU FUNCIÓN EXISTENTE para normalizar coordenadas
                normalized_bbox = normalize_bbox_coordinates(
                    full_bbox,
                    roi_coords["original_width"],
                    roi_coords["original_height"]
                )
                adjusted_detection["normalized_bbox"] = normalized_bbox

            adjusted_detections.append(adjusted_detection)

        # ✅ USA TU FUNCIÓN EXISTENTE para filtrar overlaps
        if len(adjusted_detections) > 1:
            adjusted_detections = filter_overlapping_detections(
                adjusted_detections,
                iou_threshold=0.5
            )

        logger.debug(f"🎯 Ajustadas {len(adjusted_detections)} detecciones de ROI a imagen completa")
        return adjusted_detections

    def visualize_roi(self, image: np.ndarray, detections: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Crea una visualización del ROI sobre la imagen
        NUEVA FUNCIÓN con visualización mejorada
        """
        height, width = image.shape[:2]
        roi_coords = self.calculate_central_roi((height, width))

        # Copiar imagen
        viz_image = image.copy()

        # Crear overlay semi-transparente para área no-ROI
        overlay = viz_image.copy()

        # Oscurecer área fuera del ROI
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)

        # Área del ROI transparente
        cv2.rectangle(
            overlay,
            (roi_coords["x_start"], roi_coords["y_start"]),
            (roi_coords["x_end"], roi_coords["y_end"]),
            (255, 255, 255),
            -1
        )

        # Aplicar overlay con transparencia
        alpha = 0.3  # 30% de transparencia
        cv2.addWeighted(viz_image, 1 - alpha, overlay, alpha, 0, viz_image)

        # Dibujar borde del ROI
        cv2.rectangle(
            viz_image,
            (roi_coords["x_start"], roi_coords["y_start"]),
            (roi_coords["x_end"], roi_coords["y_end"]),
            (0, 255, 255),  # Amarillo brillante
            3
        )

        # Agregar información del ROI
        info_text = [
            f"ROI Central: {self.roi_percentage}%",
            f"Tamaño: {roi_coords['width']}x{roi_coords['height']}",
            f"Posición: ({roi_coords['x_start']}, {roi_coords['y_start']})"
        ]

        y_offset = roi_coords["y_start"] - 60
        for i, text in enumerate(info_text):
            y_pos = max(20, y_offset + (i * 20))

            # Fondo para el texto
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                viz_image,
                (roi_coords["x_start"], y_pos - 15),
                (roi_coords["x_start"] + text_size[0] + 10, y_pos + 5),
                (0, 0, 0),
                -1
            )

            # Texto
            cv2.putText(
                viz_image,
                text,
                (roi_coords["x_start"] + 5, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        # Dibujar detecciones si se proporcionan
        if detections:
            for i, detection in enumerate(detections):
                bbox = detection.get("plate_bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Etiqueta de detección
                    label = f"Placa {i + 1}"
                    cv2.putText(
                        viz_image,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

        return viz_image

    def get_roi_statistics(self, original_image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        NUEVA FUNCIÓN: Obtiene estadísticas del ROI
        """
        height, width = original_image_shape
        roi_coords = self.calculate_central_roi((height, width))

        original_area = width * height
        roi_area = roi_coords["width"] * roi_coords["height"]
        reduction_factor = original_area / roi_area

        return {
            "original_dimensions": {"width": width, "height": height},
            "roi_dimensions": {"width": roi_coords["width"], "height": roi_coords["height"]},
            "original_area": original_area,
            "roi_area": roi_area,
            "area_reduction_factor": round(reduction_factor, 2),
            "area_reduction_percentage": round((1 - roi_area / original_area) * 100, 1),
            "performance_improvement_estimate": f"{reduction_factor:.1f}x más rápido",
            "roi_position": {
                "top_left": (roi_coords["x_start"], roi_coords["y_start"]),
                "bottom_right": (roi_coords["x_end"], roi_coords["y_end"]),
                "center": (roi_coords["center_x"], roi_coords["center_y"])
            }
        }