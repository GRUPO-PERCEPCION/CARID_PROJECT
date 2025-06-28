"""
Filtros específicos para validación de placas - CORREGIDO para modelos sin guión
"""
import re
from typing import Dict, Any, List
from loguru import logger
from core.utils import clean_plate_text  # ✅ USA TUS FUNCIONES


class PlateValidator:
    """Validador ajustado para modelos que detectan exactamente 6 caracteres SIN guión"""

    def __init__(self):
        self.raw_patterns = [
            r'^[A-Z]{3}\d{3}$',  # ABC123 (3 letras + 3 números)
            r'^[A-Z]{2}\d{4}$',  # AB1234 (2 letras + 4 números)
            r'^[A-Z]\d[A-Z]\d{3}$',  # ✅ NUEVO: T2C764 (letra-número-letra-3números)
            r'^[A-Z]\d{2}[A-Z]\d{2}$',  # ✅ NUEVO: A12B34 (letra-2números-letra-2números)
        ]

        logger.info("🔍 PlateValidator ajustado para modelos de 6 caracteres SIN guión")

    def validate_six_characters_only(self, plate_text: str) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Valida que la placa tenga exactamente 6 caracteres ALFANUMÉRICOS
        El modelo NO detecta guiones, solo caracteres
        """
        if not plate_text:
            return {
                "is_valid": False,
                "reason": "Texto vacío",
                "clean_text": "",
                "formatted_text": "",
                "char_count": 0,
                "model_expectation": "6_chars_no_dash"
            }

        # 🔧 LIMPIAR texto: remover espacios y caracteres especiales
        clean_text = ''.join(c for c in plate_text if c.isalnum()).upper()
        char_count = len(clean_text)

        # ✅ VERIFICAR: Exactamente 6 caracteres alfanuméricos
        if char_count != 6:
            return {
                "is_valid": False,
                "reason": f"Debe tener exactamente 6 caracteres alfanuméricos, tiene {char_count}",
                "clean_text": clean_text,
                "formatted_text": "",
                "char_count": char_count,
                "model_expectation": "6_chars_no_dash"
            }

        # ✅ VERIFICAR patrones válidos (SIN guión)
        matches_pattern = any(re.match(pattern, clean_text) for pattern in self.raw_patterns)

        if not matches_pattern:
            return {
                "is_valid": False,
                "reason": f"No coincide con patrones peruanos válidos: {clean_text}",
                "clean_text": clean_text,
                "formatted_text": "",
                "char_count": char_count,
                "expected_patterns": ["ABC123", "AB1234"],
                "model_expectation": "6_chars_no_dash"
            }

        # ✅ FORMATEAR con guión automáticamente
        formatted_text = self._add_dash_to_plate(clean_text)

        return {
            "is_valid": True,
            "reason": "Formato válido de 6 caracteres (guión agregado automáticamente)",
            "clean_text": clean_text,  # Sin guión (como detecta el modelo)
            "formatted_text": formatted_text,  # Con guión (formato final)
            "char_count": char_count,
            "pattern_matched": "6_chars_auto_dash",
            "model_expectation": "6_chars_no_dash"
        }

    def _add_dash_to_plate(self, clean_text: str) -> str:
        """
        ✅ NUEVO: Agrega guión automáticamente según patrones peruanos
        """
        if len(clean_text) != 6:
            return clean_text

        # ABC123 -> ABC-123 (3 letras + 3 números)
        if clean_text[:3].isalpha() and clean_text[3:].isdigit():
            return f"{clean_text[:3]}-{clean_text[3:]}"

        # AB1234 -> AB-1234 (2 letras + 4 números)
        elif clean_text[:2].isalpha() and clean_text[2:].isdigit():
            return f"{clean_text[:2]}-{clean_text[2:]}"

        # Si no coincide con patrones conocidos, devolver sin guión
        else:
            logger.warning(f"⚠️ Patrón no reconocido para agregar guión: {clean_text}")
            return clean_text

    def filter_detections_by_six_chars(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ✅ ACTUALIZADO: Filtra detecciones y formatea automáticamente
        """
        valid_detections = []

        for detection in detections:
            plate_text = detection.get("plate_text", "")
            validation = self.validate_six_characters_only(plate_text)

            if validation["is_valid"]:
                # ✅ USAR TEXTO FORMATEADO (con guión)
                detection["plate_text"] = validation["formatted_text"]
                detection["raw_plate_text"] = validation["clean_text"]  # Original sin guión
                detection["is_valid_format"] = True
                detection["six_char_validated"] = True
                detection["validation_info"] = validation
                detection["auto_formatted"] = True  # ✅ MARCADOR
                valid_detections.append(detection)

                logger.info(f"✅ Placa válida (6 chars): '{validation['clean_text']}' -> '{validation['formatted_text']}'")
            else:
                logger.debug(f"❌ Placa rechazada: '{plate_text}' - {validation['reason']}")

        return valid_detections

    def get_validation_stats(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        ✅ ACTUALIZADO: Estadísticas ajustadas para 6 caracteres
        """
        total = len(detections)
        valid_count = 0
        char_count_distribution = {}
        pattern_distribution = {"ABC123": 0, "AB1234": 0, "otros": 0}

        for detection in detections:
            plate_text = detection.get("plate_text", "")
            validation = self.validate_six_characters_only(plate_text)

            if validation["is_valid"]:
                valid_count += 1

                # Determinar patrón
                clean_text = validation["clean_text"]
                if len(clean_text) == 6:
                    if clean_text[:3].isalpha() and clean_text[3:].isdigit():
                        pattern_distribution["ABC123"] += 1
                    elif clean_text[:2].isalpha() and clean_text[2:].isdigit():
                        pattern_distribution["AB1234"] += 1
                    else:
                        pattern_distribution["otros"] += 1

            # Estadísticas de distribución de caracteres
            char_count = validation["char_count"]
            char_count_distribution[char_count] = char_count_distribution.get(char_count, 0) + 1

        return {
            "total_detections": total,
            "valid_six_char_detections": valid_count,
            "validation_rate": (valid_count / total * 100) if total > 0 else 0,
            "char_count_distribution": char_count_distribution,
            "pattern_distribution": pattern_distribution,
            "most_common_char_count": max(char_count_distribution,
                                        key=char_count_distribution.get) if char_count_distribution else 0,
            "expected_char_count": 6,  # ✅ ESPERAMOS EXACTAMENTE 6
            "auto_dash_formatting": True  # ✅ FORMATEAMOS AUTOMÁTICAMENTE
        }

    def format_plate_for_display(self, raw_plate_text: str) -> str:
        """
        ✅ NUEVO: Formatea placa para mostrar (agrega guión si no lo tiene)
        """
        validation = self.validate_six_characters_only(raw_plate_text)
        if validation["is_valid"]:
            return validation["formatted_text"]
        else:
            return raw_plate_text

    def is_raw_plate_valid(self, raw_plate_text: str) -> bool:
        """
        ✅ NUEVO: Verifica si una placa cruda (6 chars) es válida
        """
        validation = self.validate_six_characters_only(raw_plate_text)
        return validation["is_valid"]