"""
Filtros específicos para validación de placas - UTILIZA tu utils.py existente
"""
import re
from typing import Dict, Any, List
from loguru import logger
from core.utils import validate_peruvian_plate_format, clean_plate_text  # ✅ USA TUS FUNCIONES


class PlateValidator:
    """Validador avanzado que usa tu lógica existente + nuevas validaciones"""

    def __init__(self):
        # Patrones específicos de 6 caracteres exactos para Perú
        self.six_char_patterns = [
            r'^[A-Z]{3}\d{3}$',  # ABC123 (nuevo formato)
            r'^[A-Z]{3}-\d{3}$',  # ABC-123 (con guión)
            r'^[A-Z]{2}\d{4}$',  # AB1234 (formato anterior)
            r'^[A-Z]{2}-\d{4}$',  # AB-1234 (con guión)
        ]
        logger.info("🔍 PlateValidator inicializado con filtro de 6 caracteres")

    def validate_six_characters_only(self, plate_text: str) -> Dict[str, Any]:
        """
        NUEVA FUNCIÓN: Valida que la placa tenga exactamente 6 caracteres
        COMPLEMENTA tu función validate_peruvian_plate_format existente
        """
        if not plate_text:
            return {
                "is_valid": False,
                "reason": "Texto vacío",
                "clean_text": "",
                "char_count": 0,
                "uses_existing_validation": False
            }

        # 🔧 USA TU FUNCIÓN EXISTENTE para limpiar
        clean_text = clean_plate_text(plate_text)

        # Contar solo caracteres alfanuméricos (sin guiones)
        alnum_chars = ''.join(c for c in clean_text if c.isalnum())
        char_count = len(alnum_chars)

        # ✅ NUEVO REQUISITO: Exactamente 6 caracteres
        if char_count != 6:
            return {
                "is_valid": False,
                "reason": f"Debe tener exactamente 6 caracteres, tiene {char_count}",
                "clean_text": clean_text,
                "char_count": char_count,
                "uses_existing_validation": True
            }

        # ✅ USA TU VALIDACIÓN EXISTENTE + NUEVOS PATRONES
        existing_validation = validate_peruvian_plate_format(clean_text)

        # Verificar también los nuevos patrones de 6 caracteres
        matches_six_char_pattern = any(re.match(pattern, clean_text) for pattern in self.six_char_patterns)

        if existing_validation and matches_six_char_pattern:
            return {
                "is_valid": True,
                "reason": "Formato válido de 6 caracteres (validación existente + nueva)",
                "clean_text": clean_text,
                "char_count": char_count,
                "pattern_matched": "existing + six_char",
                "uses_existing_validation": True
            }
        elif existing_validation:
            return {
                "is_valid": False,
                "reason": "Válido según patrón existente pero no cumple requisito de 6 caracteres exactos",
                "clean_text": clean_text,
                "char_count": char_count,
                "uses_existing_validation": True
            }
        else:
            return {
                "is_valid": False,
                "reason": "No coincide con patrones válidos de placas peruanas",
                "clean_text": clean_text,
                "char_count": char_count,
                "uses_existing_validation": True
            }

    def filter_detections_by_six_chars(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        NUEVA FUNCIÓN: Filtra detecciones manteniendo solo las de exactamente 6 caracteres
        """
        valid_detections = []

        for detection in detections:
            plate_text = detection.get("plate_text", "")
            validation = self.validate_six_characters_only(plate_text)

            if validation["is_valid"]:
                # Actualizar detección con texto limpio
                detection["plate_text"] = validation["clean_text"]
                detection["is_valid_format"] = True
                detection["six_char_validated"] = True
                detection["validation_info"] = validation
                valid_detections.append(detection)

                logger.debug(f"✅ Placa válida (6 chars): {validation['clean_text']}")
            else:
                logger.debug(f"❌ Placa rechazada: {plate_text} - {validation['reason']}")

        return valid_detections

    def get_validation_stats(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        NUEVA FUNCIÓN: Obtiene estadísticas de validación
        """
        total = len(detections)
        valid_count = 0
        char_count_distribution = {}

        for detection in detections:
            plate_text = detection.get("plate_text", "")
            validation = self.validate_six_characters_only(plate_text)

            if validation["is_valid"]:
                valid_count += 1

            # Estadísticas de distribución de caracteres
            char_count = validation["char_count"]
            char_count_distribution[char_count] = char_count_distribution.get(char_count, 0) + 1

        return {
            "total_detections": total,
            "valid_six_char_detections": valid_count,
            "validation_rate": (valid_count / total * 100) if total > 0 else 0,
            "char_count_distribution": char_count_distribution,
            "most_common_char_count": max(char_count_distribution,
                                          key=char_count_distribution.get) if char_count_distribution else 0
        }