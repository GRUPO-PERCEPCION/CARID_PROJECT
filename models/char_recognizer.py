from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import re
from loguru import logger

from models.base_model import BaseModel
from config.settings import settings


class CharacterRecognizer(BaseModel):
    """
    ✅ CORREGIDO: Reconocedor ajustado para modelos que detectan exactamente 6 caracteres SIN guión
    """

    def __init__(self, model_path: Optional[str] = None):
        model_path = model_path or settings.char_model_path
        super().__init__(model_path, "Reconocedor de Caracteres")

        # ✅ CONFIGURACIONES AJUSTADAS
        self.min_char_confidence = 0.4 # Confianza mínima para caracteres
        self.expected_char_count = 6  # ✅ EXACTAMENTE 6 caracteres esperados
        self.max_characters = 8  # Máximo por si detecta ruido

        # ✅ PATRONES SIN GUIÓN (como detecta el modelo)
        self.raw_plate_patterns = [
            r'^[A-Z]{3}\d{3}$',  # ABC123 (nuevo formato sin guión)
            r'^[A-Z]{2}\d{4}$',  # AB1234 (formato anterior sin guión)
        ]

        # Patrones finales CON guión (después de formatear)
        self.formatted_plate_patterns = [
            r'^[A-Z]{3}-\d{3}$',  # ABC-123
            r'^[A-Z]{2}-\d{4}$',  # AB-1234
        ]

        logger.info("📖 CharacterRecognizer ajustado para 6 caracteres sin guión")

    def process_results(self, results: List, original_image: np.ndarray) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Procesa resultados esperando exactamente 6 caracteres
        """
        try:
            # Extraer detecciones de caracteres
            detections = self.extract_bboxes(results)

            # Filtrar caracteres válidos
            valid_chars = self._filter_valid_characters(detections)

            # Ordenar caracteres de izquierda a derecha
            sorted_chars = self._sort_characters_left_to_right(valid_chars)

            # ✅ CONSTRUIR TEXTO SIN GUIÓN (como sale del modelo)
            raw_plate_text = self._build_raw_plate_text(sorted_chars)

            # ✅ VALIDAR que tengamos exactamente 6 caracteres
            is_six_chars = len(raw_plate_text.replace(' ', '')) == 6

            # ✅ VALIDAR formato de placa (sin guión)
            is_valid_raw_format = self._validate_raw_plate_format(raw_plate_text)

            # ✅ FORMATEAR con guión automáticamente si es válido
            formatted_plate_text = ""
            if is_valid_raw_format:
                formatted_plate_text = self._add_dash_to_plate(raw_plate_text)

            # Preparar resultado
            result = {
                "success": len(sorted_chars) > 0 and is_six_chars,
                "characters_detected": len(sorted_chars),
                "plate_text": raw_plate_text,  # ✅ SIN guión (como detecta el modelo)
                "formatted_plate_text": formatted_plate_text,  # ✅ CON guión (para mostrar)
                "is_valid_format": is_valid_raw_format,
                "is_six_chars": is_six_chars,  # ✅ NUEVO
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
                    "expected_char_count": self.expected_char_count,  # ✅ NUEVO
                    "min_confidence_threshold": self.min_char_confidence,
                    "model_detects_dash": False,  # ✅ NUEVO
                    "auto_dash_formatting": True  # ✅ NUEVO
                }
            }

            if raw_plate_text:
                status = "✅ Válido" if is_valid_raw_format else "❌ Inválido"
                logger.info(f"📖 Texto reconocido: '{raw_plate_text}' {status} "
                            f"({len(raw_plate_text)} chars)")
                if formatted_plate_text:
                    logger.info(f"🔧 Texto formateado: '{formatted_plate_text}'")

            return result

        except Exception as e:
            logger.error(f"❌ Error procesando caracteres: {str(e)}")
            return {
                "success": False,
                "characters_detected": 0,
                "plate_text": "",
                "formatted_plate_text": "",
                "is_valid_format": False,
                "is_six_chars": False,
                "characters": [],
                "confidence": 0.0,
                "error": str(e)
            }

    def _filter_valid_characters(self, detections: List[Dict]) -> List[Dict[str, Any]]:
        """
        ✅ AJUSTADO: Filtra para obtener solo caracteres alfanuméricos
        """
        valid_chars = []

        for detection in detections:
            # Verificar confianza mínima
            if detection["confidence"] < self.min_char_confidence:
                continue

            # ✅ VERIFICAR que sea SOLO alfanumérico (sin guiones ni símbolos)
            char_name = detection["class_name"]
            if not self._is_valid_alphanumeric_character(char_name):
                continue

            # Agregar información adicional
            bbox = detection["bbox"]
            enhanced_char = {
                **detection,
                "character": char_name.upper(),  # ✅ ASEGURAR MAYÚSCULAS
                "center_x": (bbox[0] + bbox[2]) / 2,
                "center_y": (bbox[1] + bbox[3]) / 2,
                "width": abs(bbox[2] - bbox[0]),
                "height": abs(bbox[3] - bbox[1])
            }

            valid_chars.append(enhanced_char)

        return valid_chars

    def _is_valid_alphanumeric_character(self, char_name: str) -> bool:
        """
        ✅ NUEVO: Verifica que sea SOLO alfanumérico (A-Z, 0-9)
        El modelo NO debe detectar guiones
        """
        if len(char_name) != 1:
            return False

        # ✅ SOLO letras mayúsculas y números
        return char_name.isalnum() and (char_name.isdigit() or char_name.isupper())

    def _sort_characters_left_to_right(self, characters: List[Dict]) -> List[Dict]:
        """Ordena los caracteres de izquierda a derecha basado en center_x"""
        return sorted(characters, key=lambda x: x["center_x"])

    def _build_raw_plate_text(self, sorted_chars: List[Dict]) -> str:
        """
        ✅ NUEVO: Construye texto SIN guión (como detecta el modelo)
        """
        if not sorted_chars:
            return ""

        # ✅ EXTRAER SOLO caracteres alfanuméricos
        chars = []
        for char in sorted_chars:
            character = char["character"].upper()
            if character.isalnum():  # Solo A-Z y 0-9
                chars.append(character)

        # ✅ VERIFICAR que tengamos exactamente 6 caracteres
        raw_text = ''.join(chars)

        if len(raw_text) == self.expected_char_count:
            logger.debug(f"✅ Texto construido: '{raw_text}' ({len(raw_text)} chars)")
            return raw_text
        else:
            logger.warning(
                f"⚠️ Cantidad incorrecta de caracteres: {len(raw_text)}, esperados: {self.expected_char_count}")
            return raw_text  # Devolver anyway para debugging

    def _add_dash_to_plate(self, raw_text: str) -> str:
        """
        ✅ NUEVO: Agrega guión automáticamente según patrones peruanos
        """
        if len(raw_text) != 6:
            return raw_text

        # ABC123 -> ABC-123 (3 letras + 3 números)
        if raw_text[:3].isalpha() and raw_text[3:].isdigit():
            return f"{raw_text[:3]}-{raw_text[3:]}"

        # AB1234 -> AB-1234 (2 letras + 4 números)
        elif raw_text[:2].isalpha() and raw_text[2:].isdigit():
            return f"{raw_text[:2]}-{raw_text[2:]}"

        # Si no coincide con patrones conocidos, devolver sin guión
        else:
            logger.warning(f"⚠️ Patrón no reconocido para agregar guión: {raw_text}")
            return raw_text

    def _validate_raw_plate_format(self, plate_text: str) -> bool:
        """
        ✅ NUEVO: Valida formato de placa SIN guión (como detecta el modelo)
        """
        if not plate_text or len(plate_text) != 6:
            return False

        # ✅ VERIFICAR patrones sin guión
        for pattern in self.raw_plate_patterns:
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
        ✅ AJUSTADO: Reconoce caracteres esperando exactamente 6 sin guión
        """
        try:
            # Preprocesar imagen
            image = self.preprocess_image(image_input)

            # Realizar predicción
            logger.debug("📖 Reconociendo caracteres (esperando 6 sin guión)...")
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
                "formatted_plate_text": "",
                "is_valid_format": False,
                "is_six_chars": False,
                "characters": [],
                "confidence": 0.0,
                "error": str(e)
            }

    def get_character_details(self, image_input, **kwargs) -> List[Dict[str, Any]]:
        """
        Obtiene detalles de todos los caracteres detectados
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
                        "is_valid_alphanumeric": self._is_valid_alphanumeric_character(char["character"])
                    }
                }
                detailed_chars.append(detailed_char)

            return detailed_chars

        except Exception as e:
            logger.error(f"❌ Error obteniendo detalles de caracteres: {str(e)}")
            return []

    def visualize_characters(self, image_input, **kwargs) -> np.ndarray:
        """
        ✅ ACTUALIZADO: Visualización con información de 6 caracteres
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
                    color = (0, 0, 255)  # Azul para otros

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

            # ✅ AGREGAR INFORMACIÓN MEJORADA en la parte superior
            if result["plate_text"]:
                raw_text = result["plate_text"]
                formatted_text = result.get("formatted_plate_text", "")
                confidence = result["confidence"]
                is_six_chars = result["is_six_chars"]

                # Texto principal
                main_text = f"RAW: {raw_text} (6chars: {'✅' if is_six_chars else '❌'})"
                if formatted_text and formatted_text != raw_text:
                    main_text += f" -> {formatted_text}"

                main_text += f" (Conf: {confidence:.2f})"

                text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                cv2.rectangle(
                    result_image,
                    (10, 10),
                    (text_size[0] + 20, text_size[1] + 20),
                    (0, 0, 0),
                    -1
                )

                # Color del texto según validez
                text_color = (0, 255, 0) if result["is_valid_format"] and is_six_chars else (0, 255, 255)

                cv2.putText(
                    result_image,
                    main_text,
                    (15, text_size[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
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
        ✅ ACTUALIZADO: Post-procesa el texto para corregir errores comunes
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

        # ✅ REMOVER cualquier carácter no alfanumérico
        corrected = re.sub(r'[^A-Z0-9]', '', plate_text.upper())

        # ✅ APLICAR correcciones basadas en posición para placas de 6 caracteres
        if len(corrected) == 6:
            # Formato ABC123: las primeras 3 deben ser letras, las últimas 3 números
            if corrected[:3].isalnum() and corrected[3:].isalnum():
                # Corregir letras en posiciones de números y viceversa
                result = ""
                for i, char in enumerate(corrected):
                    if i < 3:  # Posiciones de letras
                        if char.isdigit() and char in corrections:
                            result += corrections[char]
                        else:
                            result += char
                    else:  # Posiciones de números
                        if char.isalpha() and char in corrections:
                            result += corrections[char]
                        else:
                            result += char
                corrected = result

        return corrected