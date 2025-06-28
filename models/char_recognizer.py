from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import re
from loguru import logger

from models.base_model import BaseModel
from config.settings import settings


class CharacterRecognizer(BaseModel):
    """
    ✅ ACTUALIZADO: Reconocedor con configuración centralizada para modelos de 6 caracteres SIN guión
    """

    def __init__(self, model_path: Optional[str] = None):
        model_path = model_path or settings.char_model_path
        super().__init__(model_path, "Reconocedor de Caracteres")

        # ✅ CONFIGURACIONES CENTRALIZADAS
        char_config = settings.get_char_recognizer_config()

        self.min_char_confidence = char_config['min_char_confidence']
        self.expected_char_count = char_config['expected_char_count']
        self.max_characters = char_config['max_characters']
        self.force_six_characters = char_config['force_six_characters']
        self.strict_validation = char_config['strict_validation']

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

        logger.info("📖 CharacterRecognizer inicializado con configuración centralizada")
        logger.debug(f"📊 Config aplicada: min_conf={self.min_char_confidence}, "
                     f"expected_chars={self.expected_char_count}, "
                     f"force_six={self.force_six_characters}, strict={self.strict_validation}")

    def process_results(self, results: List, original_image: np.ndarray) -> Dict[str, Any]:
        """
        ✅ ACTUALIZADO: Procesa resultados usando configuración centralizada
        """
        try:
            # Extraer detecciones de caracteres
            detections = self.extract_bboxes(results)

            # Filtrar caracteres válidos usando config centralizada
            valid_chars = self._filter_valid_characters(detections)

            # Ordenar caracteres de izquierda a derecha
            sorted_chars = self._sort_characters_left_to_right(valid_chars)

            # ✅ CONSTRUIR TEXTO SIN GUIÓN (como sale del modelo)
            raw_plate_text = self._build_raw_plate_text(sorted_chars)

            # ✅ VALIDAR usando configuración centralizada
            is_expected_count = len(raw_plate_text.replace(' ', '')) == self.expected_char_count
            is_valid_raw_format = self._validate_raw_plate_format(raw_plate_text)

            # ✅ FORMATEAR con guión automáticamente si es válido y si está habilitado
            formatted_plate_text = ""
            auto_formatted = False
            if is_valid_raw_format and self.force_six_characters:
                formatted_plate_text = self._add_dash_to_plate(raw_plate_text)
                auto_formatted = True

            # Aplicar validación estricta si está habilitada
            final_success = is_expected_count and (is_valid_raw_format or not self.strict_validation)

            # Preparar resultado
            result = {
                "success": final_success and len(sorted_chars) > 0,
                "characters_detected": len(sorted_chars),
                "plate_text": raw_plate_text,  # ✅ SIN guión (como detecta el modelo)
                "formatted_plate_text": formatted_plate_text,  # ✅ CON guión (para mostrar)
                "is_valid_format": is_valid_raw_format,
                "is_six_chars": is_expected_count,  # ✅ ACTUALIZADO usando config
                "auto_formatted": auto_formatted,  # ✅ NUEVO
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
                    "expected_char_count": self.expected_char_count,
                    "min_confidence_threshold": self.min_char_confidence,
                    "model_detects_dash": False,
                    "auto_dash_formatting": self.force_six_characters,
                    "strict_validation": self.strict_validation,
                    "configuration_source": "centralized_settings"  # ✅ NUEVO
                }
            }

            # ✅ LOG MEJORADO CON CONFIG INFO
            if raw_plate_text:
                status = "✅ Válido" if is_valid_raw_format else "❌ Inválido"
                count_status = f"({len(raw_plate_text)} chars, esperados: {self.expected_char_count})"
                logger.info(f"📖 Texto reconocido: '{raw_plate_text}' {status} {count_status}")
                if formatted_plate_text and auto_formatted:
                    logger.info(f"🔧 Auto-formateado: '{formatted_plate_text}' (config centralizada)")

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
                "auto_formatted": False,
                "characters": [],
                "confidence": 0.0,
                "error": str(e),
                "configuration_source": "centralized_settings"
            }

    def _filter_valid_characters(self, detections: List[Dict]) -> List[Dict[str, Any]]:
        """
        ✅ ACTUALIZADO: Filtra usando configuración centralizada
        """
        valid_chars = []

        for detection in detections:
            # ✅ VERIFICAR CONFIANZA MÍNIMA CENTRALIZADA
            if detection["confidence"] < self.min_char_confidence:
                logger.debug(f"🔍 Carácter descartado por confianza baja: "
                             f"{detection['confidence']:.3f} < {self.min_char_confidence}")
                continue

            # ✅ VERIFICAR que sea SOLO alfanumérico (sin guiones ni símbolos)
            char_name = detection["class_name"]
            if not self._is_valid_alphanumeric_character(char_name):
                logger.debug(f"🔍 Carácter descartado por no ser alfanumérico: '{char_name}'")
                continue

            # Agregar información adicional
            bbox = detection["bbox"]
            enhanced_char = {
                **detection,
                "character": char_name.upper(),  # ✅ ASEGURAR MAYÚSCULAS
                "center_x": (bbox[0] + bbox[2]) / 2,
                "center_y": (bbox[1] + bbox[3]) / 2,
                "width": abs(bbox[2] - bbox[0]),
                "height": abs(bbox[3] - bbox[1]),
                "validation_passed": {  # ✅ NUEVA INFO DE VALIDACIÓN
                    "min_confidence": detection["confidence"] >= self.min_char_confidence,
                    "is_alphanumeric": True,
                    "config_source": "centralized_settings"
                }
            }

            valid_chars.append(enhanced_char)

        # ✅ LIMITAR CARACTERES SEGÚN CONFIG CENTRALIZADA
        if len(valid_chars) > self.max_characters:
            logger.warning(f"⚠️ Limitando caracteres: {len(valid_chars)} -> {self.max_characters} "
                           f"(según config centralizada)")
            # Ordenar por confianza y tomar los mejores
            valid_chars.sort(key=lambda x: x["confidence"], reverse=True)
            valid_chars = valid_chars[:self.max_characters]

        return valid_chars

    def _is_valid_alphanumeric_character(self, char_name: str) -> bool:
        """
        ✅ VERIFICAR usando configuración centralizada si es necesario
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
        ✅ ACTUALIZADO: Construye texto usando configuración centralizada
        """
        if not sorted_chars:
            return ""

        # ✅ EXTRAER SOLO caracteres alfanuméricos
        chars = []
        for char in sorted_chars:
            character = char["character"].upper()
            if character.isalnum():  # Solo A-Z y 0-9
                chars.append(character)

        # ✅ VERIFICAR que tengamos la cantidad esperada según config
        raw_text = ''.join(chars)

        if len(raw_text) == self.expected_char_count:
            logger.debug(f"✅ Texto construido: '{raw_text}' "
                         f"({len(raw_text)} chars, esperados: {self.expected_char_count})")
            return raw_text
        else:
            if self.strict_validation:
                logger.warning(f"⚠️ Cantidad incorrecta de caracteres: {len(raw_text)}, "
                               f"esperados: {self.expected_char_count} (validación estricta)")
            else:
                logger.debug(f"📝 Cantidad diferente de caracteres: {len(raw_text)}, "
                             f"esperados: {self.expected_char_count} (validación permisiva)")
            return raw_text  # Devolver anyway para debugging

    def _add_dash_to_plate(self, raw_text: str) -> str:
        """
        ✅ ACTUALIZADO: Agrega guión usando configuración centralizada
        """
        if not self.force_six_characters or len(raw_text) != self.expected_char_count:
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
        ✅ ACTUALIZADO: Valida formato usando configuración centralizada
        """
        if not plate_text or (self.strict_validation and len(plate_text) != self.expected_char_count):
            return False

        # Si no es validación estricta, permitir diferentes longitudes
        if not self.strict_validation and len(plate_text) < 4:
            return False

        # ✅ VERIFICAR patrones sin guión
        for pattern in self.raw_plate_patterns:
            if re.match(pattern, plate_text):
                return True

        return False

    def _calculate_overall_confidence(self, characters: List[Dict]) -> float:
        """✅ ACTUALIZADO: Calcula confianza usando configuración centralizada"""
        if not characters:
            return 0.0

        # Si tenemos menos caracteres de los esperados, penalizar
        char_count_penalty = 1.0
        if self.strict_validation and len(characters) != self.expected_char_count:
            char_count_penalty = 0.8

        # Promedio ponderado por área del carácter
        total_weighted_conf = 0.0
        total_weight = 0.0

        for char in characters:
            # Verificar que el carácter pasó el filtro de confianza mínima
            if char["confidence"] >= self.min_char_confidence:
                area = char["width"] * char["height"]
                weight = area * char["confidence"]
                total_weighted_conf += weight
                total_weight += area

        if total_weight == 0:
            return 0.0

        base_confidence = total_weighted_conf / total_weight
        return base_confidence * char_count_penalty

    def recognize_characters(self, image_input, **kwargs) -> Dict[str, Any]:
        """
        ✅ ACTUALIZADO: Reconoce caracteres usando configuración centralizada
        """
        try:
            # Preprocesar imagen
            image = self.preprocess_image(image_input)

            # ✅ APLICAR CONFIGURACIÓN CENTRALIZADA
            model_kwargs = self._build_recognition_kwargs(kwargs)

            # Realizar predicción
            logger.debug(f"📖 Reconociendo caracteres con config centralizada: {model_kwargs}")
            results = self.predict(image, **model_kwargs)

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
                "auto_formatted": False,
                "characters": [],
                "confidence": 0.0,
                "error": str(e),
                "configuration_source": "centralized_settings"
            }

    def _build_recognition_kwargs(self, user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """✅ NUEVO: Construye kwargs usando configuración centralizada como fallback"""

        # Usar config de settings como base
        recognition_kwargs = {
            'conf': user_kwargs.get('conf', self.min_char_confidence),
            'verbose': user_kwargs.get('verbose', False)
        }

        logger.debug(f"🔧 Config base CharRecognizer: min_conf={self.min_char_confidence}")
        logger.debug(f"📝 User kwargs: {user_kwargs}")
        logger.debug(f"⚙️ Recognition kwargs finales: {recognition_kwargs}")

        return recognition_kwargs

    def get_character_details(self, image_input, **kwargs) -> List[Dict[str, Any]]:
        """
        ✅ ACTUALIZADO: Obtiene detalles de todos los caracteres detectados usando config centralizada
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
                        "is_valid_alphanumeric": self._is_valid_alphanumeric_character(char["character"]),
                        "meets_min_confidence": char["confidence"] >= self.min_char_confidence,
                        "config_source": "centralized_settings"
                    },
                    "config_info": {
                        "expected_total_chars": self.expected_char_count,
                        "min_confidence_threshold": self.min_char_confidence,
                        "strict_validation": self.strict_validation
                    }
                }
                detailed_chars.append(detailed_char)

            return detailed_chars

        except Exception as e:
            logger.error(f"❌ Error obteniendo detalles de caracteres: {str(e)}")
            return []

    def visualize_characters(self, image_input, **kwargs) -> np.ndarray:
        """
        ✅ ACTUALIZADO: Visualización con información de configuración centralizada
        """
        try:
            # Reconocer caracteres
            result = self.recognize_characters(image_input, **kwargs)

            # Cargar imagen original
            image = self.preprocess_image(image_input)
            result_image = image.copy()

            if not result["success"]:
                # ✅ AGREGAR INFO DE CONFIG INCLUSO SI NO HAY RESULTADOS
                self._add_config_info_to_image(result_image, result)
                return result_image

            # ✅ AGREGAR INFORMACIÓN DE CONFIG EN LA VISUALIZACIÓN
            self._add_config_info_to_image(result_image, result)

            # Dibujar cada carácter
            for i, char in enumerate(result["characters"]):
                bbox = char["bbox"]
                confidence = char["confidence"]
                character = char["character"]

                # Coordenadas del rectángulo
                x1, y1, x2, y2 = map(int, bbox)

                # ✅ COLOR BASADO EN VALIDACIÓN CENTRALIZADA
                validation = char.get("validation_passed", {})
                if validation.get("min_confidence", False):
                    if character.isalpha():
                        color = (0, 255, 0)  # Verde para letras válidas
                    elif character.isdigit():
                        color = (255, 0, 0)  # Rojo para números válidos
                    else:
                        color = (0, 0, 255)  # Azul para otros válidos
                else:
                    color = (128, 128, 128)  # Gris para baja confianza

                # Dibujar rectángulo
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

                # ✅ TEXTO CON INFORMACIÓN DE CONFIG
                label = f"{character} ({confidence:.2f})"
                if confidence < self.min_char_confidence:
                    label += " [LOW]"

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

            return result_image

        except Exception as e:
            logger.error(f"❌ Error visualizando caracteres: {str(e)}")
            return self.preprocess_image(image_input)

    def _add_config_info_to_image(self, image: np.ndarray, result: Dict[str, Any]):
        """✅ NUEVO: Agrega información de configuración a la imagen"""
        try:
            # Información principal
            raw_text = result.get("plate_text", "")
            formatted_text = result.get("formatted_plate_text", "")
            confidence = result.get("confidence", 0.0)
            is_expected_count = result.get("is_six_chars", False)
            auto_formatted = result.get("auto_formatted", False)

            # ✅ TEXTO PRINCIPAL CON INFO DE CONFIG
            if raw_text:
                main_text = f"RAW: {raw_text} ({len(raw_text)}/{self.expected_char_count})"
                if is_expected_count:
                    main_text += " ✅"
                else:
                    main_text += " ❌"

                if formatted_text and formatted_text != raw_text:
                    main_text += f" -> {formatted_text}"
                    if auto_formatted:
                        main_text += " 🔧"

                main_text += f" (Conf: {confidence:.2f})"
            else:
                main_text = f"Sin texto (min_conf: {self.min_char_confidence}, esperados: {self.expected_char_count})"

            text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

            cv2.rectangle(
                image,
                (10, 10),
                (text_size[0] + 20, text_size[1] + 50),  # Más espacio para info adicional
                (0, 0, 0),
                -1
            )

            # Color del texto según validez
            text_color = (0, 255, 0) if result.get("is_valid_format", False) and is_expected_count else (0, 255, 255)

            cv2.putText(
                image,
                main_text,
                (15, text_size[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2
            )

            # ✅ INFORMACIÓN DE CONFIGURACIÓN
            config_text = f"Config: min_conf={self.min_char_confidence}, strict={self.strict_validation}, force_6={self.force_six_characters}"
            cv2.putText(
                image,
                config_text,
                (15, text_size[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

        except Exception as e:
            logger.debug(f"Error agregando info de config: {e}")

    def enhance_image_for_recognition(self, image: np.ndarray) -> np.ndarray:
        """
        ✅ ACTUALIZADO: Mejora la imagen usando configuración centralizada si es necesario
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # ✅ APLICAR FILTROS SEGÚN CONFIG (se podría agregar config para esto)
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
        ✅ ACTUALIZADO: Post-procesa el texto usando configuración centralizada
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

        # ✅ APLICAR correcciones basadas en posición usando config centralizada
        if len(corrected) == self.expected_char_count and self.force_six_characters:
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

    # ✅ MÉTODOS PARA CONFIGURACIÓN DINÁMICA

    def get_current_config(self) -> Dict[str, Any]:
        """✅ NUEVO: Obtiene la configuración actual del reconocedor"""
        return {
            "min_char_confidence": self.min_char_confidence,
            "expected_char_count": self.expected_char_count,
            "max_characters": self.max_characters,
            "force_six_characters": self.force_six_characters,
            "strict_validation": self.strict_validation,
            "raw_plate_patterns": self.raw_plate_patterns,
            "formatted_plate_patterns": self.formatted_plate_patterns,
            "configuration_source": "centralized_settings"
        }

    def update_config_from_settings(self):
        """✅ NUEVO: Recarga la configuración desde settings"""
        char_config = settings.get_char_recognizer_config()

        self.min_char_confidence = char_config['min_char_confidence']
        self.expected_char_count = char_config['expected_char_count']
        self.max_characters = char_config['max_characters']
        self.force_six_characters = char_config['force_six_characters']
        self.strict_validation = char_config['strict_validation']

        logger.info("🔄 Configuración del CharacterRecognizer recargada desde settings")

    def validate_recognition_params(self, **kwargs) -> Dict[str, Any]:
        """✅ NUEVO: Valida parámetros de reconocimiento"""
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "config_applied": self.get_current_config()
        }

        try:
            conf = kwargs.get('conf', self.min_char_confidence)

            # Validaciones básicas
            if conf < 0.1 or conf > 1.0:
                validation["errors"].append("confidence debe estar entre 0.1 y 1.0")

            # Advertencias usando config centralizada
            if conf < self.min_char_confidence:
                validation["warnings"].append(
                    f"confidence ({conf}) menor que mínimo recomendado ({self.min_char_confidence})")

            if conf > 0.9:
                validation["warnings"].append("confidence muy alto puede rechazar caracteres válidos")

            validation["is_valid"] = len(validation["errors"]) == 0

        except Exception as e:
            validation["is_valid"] = False
            validation["errors"].append(f"Error validando parámetros: {str(e)}")

        return validation