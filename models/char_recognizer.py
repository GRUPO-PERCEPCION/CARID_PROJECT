from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import re
from loguru import logger

from models.base_model import BaseModel
from config.settings import settings


class CharacterRecognizer(BaseModel):
    """Reconocedor de caracteres en placas vehiculares usando YOLOv8"""

    def __init__(self, model_path: Optional[str] = None):
        model_path = model_path or settings.char_model_path
        super().__init__(model_path, "Reconocedor de Caracteres")

        # Configuraciones específicas para reconocimiento de caracteres
        self.min_char_confidence = 0.3  # Confianza mínima para caracteres
        self.max_characters = 10  # Máximo número de caracteres por placa

        # Patrones de placas peruanas
        self.plate_patterns = [
            r'^[A-Z]{3}-\d{3}$',  # Formato actual: ABC-123
            r'^[A-Z]{2}-\d{4}$',  # Formato anterior: AB-1234
            r'^[A-Z]\d{2}-\d{3}$',  # Motos: A12-345
            r'^[A-Z]{3}\d{3}$',  # Sin guión: ABC123
        ]

    def process_results(self, results: List, original_image: np.ndarray) -> Dict[str, Any]:
        """Procesa los resultados de reconocimiento de caracteres"""
        try:
            # Extraer detecciones de caracteres
            detections = self.extract_bboxes(results)

            # Filtrar caracteres válidos
            valid_chars = self._filter_valid_characters(detections)

            # Ordenar caracteres de izquierda a derecha
            sorted_chars = self._sort_characters_left_to_right(valid_chars)

            # Construir texto de la placa
            plate_text = self._build_plate_text(sorted_chars)

            # Validar formato de placa
            is_valid_format = self._validate_plate_format(plate_text)

            # Preparar resultado
            result = {
                "success": len(sorted_chars) > 0,
                "characters_detected": len(sorted_chars),
                "plate_text": plate_text,
                "is_valid_format": is_valid_format,
                "characters": sorted_chars,
                "confidence": self._calculate_overall_confidence(sorted_chars),
                "image_shape": {
                    "height": original_image.shape[0],
                    "width": original_image.shape[1],
                    "channels": original_image.shape[2] if len(original_image.shape) > 2 else 1
                },
                "processing_info": {
                    "total_detections": len(detections),
                    "valid_characters": len(valid_chars),
                    "min_confidence_threshold": self.min_char_confidence
                }
            }

            logger.info(f"📖 Texto reconocido: '{plate_text}' ({'Válido' if is_valid_format else 'Inválido'})")
            return result

        except Exception as e:
            logger.error(f"❌ Error procesando caracteres: {str(e)}")
            return {
                "success": False,
                "characters_detected": 0,
                "plate_text": "",
                "is_valid_format": False,
                "characters": [],
                "confidence": 0.0,
                "error": str(e)
            }

    def _filter_valid_characters(self, detections: List[Dict]) -> List[Dict[str, Any]]:
        """Filtra detecciones para obtener solo caracteres válidos"""
        valid_chars = []

        for detection in detections:
            # Verificar confianza mínima
            if detection["confidence"] < self.min_char_confidence:
                continue

            # Verificar que el nombre de clase sea un carácter válido
            char_name = detection["class_name"]
            if not self._is_valid_character(char_name):
                continue

            # Agregar información adicional
            bbox = detection["bbox"]
            enhanced_char = {
                **detection,
                "character": char_name,
                "center_x": (bbox[0] + bbox[2]) / 2,
                "center_y": (bbox[1] + bbox[3]) / 2,
                "width": abs(bbox[2] - bbox[0]),
                "height": abs(bbox[3] - bbox[1])
            }

            valid_chars.append(enhanced_char)

        return valid_chars

    def _is_valid_character(self, char_name: str) -> bool:
        """Verifica si el carácter es válido para placas peruanas"""
        # Letras válidas: A-Z
        # Números válidos: 0-9
        # Caracteres especiales: - (guión)

        if len(char_name) != 1:
            return char_name in ['-', 'dash', 'hyphen']

        return char_name.isalnum() and (char_name.isdigit() or char_name.isupper())

    def _sort_characters_left_to_right(self, characters: List[Dict]) -> List[Dict]:
        """Ordena los caracteres de izquierda a derecha basado en center_x"""
        return sorted(characters, key=lambda x: x["center_x"])

    def _build_plate_text(self, sorted_chars: List[Dict]) -> str:
        """Construye el texto de la placa a partir de caracteres ordenados"""
        if not sorted_chars:
            return ""

        # Extraer caracteres
        chars = [char["character"] for char in sorted_chars]

        # Limpiar caracteres especiales
        cleaned_chars = []
        for char in chars:
            if char in ['-', 'dash', 'hyphen']:
                cleaned_chars.append('-')
            else:
                cleaned_chars.append(char.upper())

        # Construir texto base
        base_text = ''.join(cleaned_chars)

        # Intentar formatear según patrones peruanos
        formatted_text = self._format_peruvian_plate(base_text)

        return formatted_text

    def _format_peruvian_plate(self, text: str) -> str:
        """Intenta formatear el texto según los patrones de placas peruanas"""
        # Remover caracteres no válidos
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

        if len(clean_text) < 5:
            return text  # Muy corto, retornar original

        # Patrones comunes
        if len(clean_text) == 6:
            # Formato ABC123 -> ABC-123
            if clean_text[:3].isalpha() and clean_text[3:].isdigit():
                return f"{clean_text[:3]}-{clean_text[3:]}"
            # Formato AB1234 -> AB-1234
            elif clean_text[:2].isalpha() and clean_text[2:].isdigit():
                return f"{clean_text[:2]}-{clean_text[2:]}"

        elif len(clean_text) == 5:
            # Formato A1234 -> A12-345 (motos)
            if clean_text[0].isalpha() and clean_text[1:].isdigit():
                return f"{clean_text[:3]}-{clean_text[3:]}"

        # Si ya tiene el formato correcto, mantenerlo
        for pattern in self.plate_patterns:
            if re.match(pattern, text):
                return text

        return text  # Retornar original si no se puede formatear

    def _validate_plate_format(self, plate_text: str) -> bool:
        """Valida si el texto coincide con algún patrón de placa peruana"""
        if not plate_text:
            return False

        for pattern in self.plate_patterns:
            if re.match(pattern, plate_text):
                return True

        return False

    def _calculate_overall_confidence(self, characters: List[Dict]) -> float:
        """Calcula la confianza general basada en todos los caracteres"""
        if not characters:
            return 0.0

        # Promedio ponderado por área del carácter
        total_weighted_conf = 0.0
        total_weight = 0.0

        for char in characters:
            area = char["width"] * char["height"]
            weight = area * char["confidence"]  # Peso = área * confianza
            total_weighted_conf += weight
            total_weight += area

        if total_weight == 0:
            return 0.0

        return total_weighted_conf / total_weight

    def recognize_characters(self, image_input, **kwargs) -> Dict[str, Any]:
        """
        Reconoce caracteres en una imagen de placa

        Args:
            image_input: Puede ser path de imagen (str) o numpy array
            **kwargs: Parámetros adicionales para la predicción

        Returns:
            Dict con resultados de reconocimiento
        """
        try:
            # Preprocesar imagen
            image = self.preprocess_image(image_input)

            # Realizar predicción
            logger.info("📖 Reconociendo caracteres...")
            results = self.predict(image, **kwargs)

            # Procesar resultados
            processed_results = self.process_results(results, image)

            return processed_results

        except Exception as e:
            logger.error(f"❌ Error en reconocimiento de caracteres: {str(e)}")
            return {
                "success": False,
                "characters_detected": 0,
                "plate_text": "",
                "is_valid_format": False,
                "characters": [],
                "confidence": 0.0,
                "error": str(e)
            }

    def get_character_details(self, image_input, **kwargs) -> List[Dict[str, Any]]:
        """
        Obtiene detalles de todos los caracteres detectados

        Returns:
            Lista con información detallada de cada carácter
        """
        try:
            result = self.recognize_characters(image_input, **kwargs)

            if not result["success"]:
                return []

            # Agregar información adicional a cada carácter
            detailed_chars = []
            for char in result["characters"]:
                detailed_char = {
                    **char,
                    "position_info": {
                        "is_first": char == result["characters"][0],
                        "is_last": char == result["characters"][-1],
                        "index": result["characters"].index(char)
                    },
                    "validation": {
                        "is_letter": char["character"].isalpha(),
                        "is_digit": char["character"].isdigit(),
                        "is_valid_for_peru": self._is_valid_character(char["character"])
                    }
                }
                detailed_chars.append(detailed_char)

            return detailed_chars

        except Exception as e:
            logger.error(f"❌ Error obteniendo detalles de caracteres: {str(e)}")
            return []

    def visualize_characters(self, image_input, **kwargs) -> np.ndarray:
        """
        Crea una imagen con los caracteres reconocidos visualizados

        Returns:
            Imagen con bounding boxes y texto de caracteres
        """
        try:
            # Reconocer caracteres
            result = self.recognize_characters(image_input, **kwargs)

            # Cargar imagen original
            image = self.preprocess_image(image_input)
            result_image = image.copy()

            if not result["success"]:
                return result_image

            # Dibujar cada carácter
            for i, char in enumerate(result["characters"]):
                bbox = char["bbox"]
                confidence = char["confidence"]
                character = char["character"]

                # Coordenadas del rectángulo
                x1, y1, x2, y2 = map(int, bbox)

                # Color basado en tipo de carácter
                if character.isalpha():
                    color = (0, 255, 0)  # Verde para letras
                elif character.isdigit():
                    color = (255, 0, 0)  # Rojo para números
                else:
                    color = (0, 0, 255)  # Azul para símbolos

                # Dibujar rectángulo
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

                # Texto con carácter y confianza
                label = f"{character} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                # Fondo para el texto
                cv2.rectangle(
                    result_image,
                    (x1, y1 - label_size[1] - 5),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )

                # Texto
                cv2.putText(
                    result_image,
                    label,
                    (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

                # Número de orden
                cv2.putText(
                    result_image,
                    str(i + 1),
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    2
                )

            # Agregar texto completo en la parte superior
            if result["plate_text"]:
                plate_text = result["plate_text"]
                confidence = result["confidence"]

                # Fondo para texto principal
                text = f"PLACA: {plate_text} (Conf: {confidence:.2f})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

                cv2.rectangle(
                    result_image,
                    (10, 10),
                    (text_size[0] + 20, text_size[1] + 20),
                    (0, 0, 0),
                    -1
                )

                # Color del texto según validez
                text_color = (0, 255, 0) if result["is_valid_format"] else (0, 255, 255)

                cv2.putText(
                    result_image,
                    text,
                    (15, text_size[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    text_color,
                    2
                )

            return result_image

        except Exception as e:
            logger.error(f"❌ Error visualizando caracteres: {str(e)}")
            return self.preprocess_image(image_input)

    def enhance_image_for_recognition(self, image: np.ndarray) -> np.ndarray:
        """
        Mejora la imagen para un mejor reconocimiento de caracteres

        Args:
            image: Imagen de entrada (numpy array)

        Returns:
            Imagen mejorada
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # Aplicar filtros para mejorar contraste
            # 1. Ecualización de histograma adaptativa
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # 2. Reducción de ruido
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # 3. Sharpening
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            # Convertir de vuelta a 3 canales si es necesario
            if len(image.shape) == 3:
                enhanced_image = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
            else:
                enhanced_image = sharpened

            return enhanced_image

        except Exception as e:
            logger.error(f"❌ Error mejorando imagen: {str(e)}")
            return image

    def post_process_text(self, plate_text: str) -> str:
        """
        Post-procesa el texto reconocido para corregir errores comunes

        Args:
            plate_text: Texto de placa reconocido

        Returns:
            Texto corregido
        """
        if not plate_text:
            return plate_text

        # Correcciones comunes OCR -> Caracter correcto
        corrections = {
            '0': 'O',  # En contexto de letras
            'O': '0',  # En contexto de números
            '1': 'I',  # En contexto de letras
            'I': '1',  # En contexto de números
            '5': 'S',  # En contexto de letras
            'S': '5',  # En contexto de números
            '8': 'B',  # En contexto de letras
            'B': '8',  # En contexto de números
        }

        # Aplicar correcciones basadas en posición
        corrected = plate_text

        # Lógica específica para placas peruanas
        # Formato ABC-123: las primeras 3 deben ser letras, las últimas 3 números

        # Remover espacios y caracteres extraños
        corrected = re.sub(r'[^A-Z0-9\-]', '', corrected.upper())

        return corrected