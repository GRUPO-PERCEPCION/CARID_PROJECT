from typing import Optional, Dict, Any, List, Tuple
import torch
import os
import numpy as np
from loguru import logger
from config.settings import settings


class ModelManager:
    """Gestor centralizado y mejorado para los modelos YOLOv8"""

    def __init__(self):
        # Instancias de los modelos
        self.plate_detector = None
        self.char_recognizer = None

        # Estado del gestor
        self.device = settings.device
        self.is_loaded = False
        self.models_info = {}

        # Configurar logger
        logger.info(f"🚀 Inicializando ModelManager con dispositivo: {self.device}")

    def load_models(self) -> bool:
        """Carga ambos modelos YOLOv8"""
        try:
            logger.info("🤖 Iniciando carga de modelos...")

            # Verificar disponibilidad de CUDA
            self._check_cuda_availability()

            # Cargar detector de placas
            if not self._load_plate_detector():
                return False

            # Cargar reconocedor de caracteres
            if not self._load_character_recognizer():
                return False

            # Actualizar estado
            self.is_loaded = True
            self._update_models_info()

            logger.success("✅ Ambos modelos cargados exitosamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error al cargar modelos: {str(e)}")
            self.is_loaded = False
            return False

    def _check_cuda_availability(self):
        """Verifica y reporta la disponibilidad de CUDA"""
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

            logger.info(f"🚀 CUDA disponible: {cuda_version}")
            logger.info(f"🔥 GPU: {gpu_name}")
            logger.info(f"💾 Memoria GPU: {gpu_memory:.1f} GB")
        else:
            logger.warning("⚠️ CUDA no disponible, usando CPU")
            self.device = "cpu"

    def _load_plate_detector(self) -> bool:
        """Carga el detector de placas"""
        try:
            logger.info("📦 Cargando detector de placas...")

            # Importación diferida para evitar dependencias circulares
            from models.plate_detector import PlateDetector

            # Crear instancia del detector
            self.plate_detector = PlateDetector()

            # Cargar el modelo
            if not self.plate_detector.load_model():
                logger.error("❌ Error cargando detector de placas")
                return False

            logger.success("✅ Detector de placas cargado")
            return True

        except Exception as e:
            logger.error(f"❌ Error en detector de placas: {str(e)}")
            return False

    def _load_character_recognizer(self) -> bool:
        """Carga el reconocedor de caracteres"""
        try:
            logger.info("📦 Cargando reconocedor de caracteres...")

            # Importación diferida para evitar dependencias circulares
            from models.char_recognizer import CharacterRecognizer

            # Crear instancia del reconocedor
            self.char_recognizer = CharacterRecognizer()

            # Cargar el modelo
            if not self.char_recognizer.load_model():
                logger.error("❌ Error cargando reconocedor de caracteres")
                return False

            logger.success("✅ Reconocedor de caracteres cargado")
            return True

        except Exception as e:
            logger.error(f"❌ Error en reconocedor de caracteres: {str(e)}")
            return False

    def _update_models_info(self):
        """Actualiza la información de los modelos"""
        self.models_info = {
            "plate_detector": self.plate_detector.get_model_info() if self.plate_detector else None,
            "char_recognizer": self.char_recognizer.get_model_info() if self.char_recognizer else None,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "models_loaded": self.is_loaded
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información detallada sobre los modelos cargados"""
        base_info = {
            "models_loaded": self.is_loaded,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "plate_detector_loaded": self.plate_detector is not None and self.plate_detector.is_loaded,
            "char_recognizer_loaded": self.char_recognizer is not None and self.char_recognizer.is_loaded,
            "confidence_threshold": settings.model_confidence_threshold,
            "iou_threshold": settings.model_iou_threshold
        }

        # Agregar información específica de cada modelo
        if self.models_info:
            base_info.update(self.models_info)

        return base_info

    def warmup_models(self):
        """Realiza warmup de ambos modelos"""
        try:
            if not self.is_loaded:
                logger.warning("⚠️ Modelos no cargados, saltando warmup")
                return

            logger.info("🔥 Realizando warmup de modelos...")

            # Warmup detector de placas
            if self.plate_detector:
                self.plate_detector.warmup()

            # Warmup reconocedor de caracteres
            if self.char_recognizer:
                self.char_recognizer.warmup()

            logger.success("✅ Warmup de ambos modelos completado")

        except Exception as e:
            logger.warning(f"⚠️ Error en warmup: {str(e)}")

    # ===== MÉTODOS DE PROCESAMIENTO PRINCIPAL =====

    def process_full_pipeline(self, image_input, **kwargs) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo: detección de placas + reconocimiento de caracteres

        Args:
            image_input: Imagen de entrada (path o numpy array)
            **kwargs: Parámetros adicionales

        Returns:
            Resultado completo del procesamiento
        """
        try:
            if not self.is_loaded:
                raise ValueError("Modelos no están cargados")

            logger.info("🔄 Iniciando pipeline completo...")

            # Paso 1: Detectar placas
            logger.info("🎯 Paso 1: Detectando placas...")
            plate_results = self.plate_detector.detect_plates(image_input, **kwargs)

            if not plate_results["success"] or plate_results["plates_detected"] == 0:
                return {
                    "success": False,
                    "message": "No se detectaron placas en la imagen",
                    "plate_detection": plate_results,
                    "character_recognition": None,
                    "final_results": []
                }

            # Paso 2: Procesar cada placa detectada
            logger.info(f"📖 Paso 2: Reconociendo caracteres en {plate_results['plates_detected']} placa(s)...")

            final_results = []

            for i, plate_info in enumerate(plate_results["plates"]):
                try:
                    # Extraer región de la placa
                    image = self.plate_detector.preprocess_image(image_input)
                    plate_region = self.plate_detector.crop_image_from_bbox(
                        image,
                        plate_info["bbox"],
                        padding=10
                    )

                    # Reconocer caracteres en la región
                    char_results = self.char_recognizer.recognize_characters(
                        plate_region,
                        **kwargs
                    )

                    # Combinar resultados
                    combined_result = {
                        "plate_id": i + 1,
                        "plate_bbox": plate_info["bbox"],
                        "plate_confidence": plate_info["confidence"],
                        "plate_area": plate_info["area"],
                        "character_recognition": char_results,
                        "plate_text": char_results.get("plate_text", ""),
                        "overall_confidence": self._calculate_combined_confidence(
                            plate_info["confidence"],
                            char_results.get("confidence", 0.0)
                        ),
                        "is_valid_plate": char_results.get("is_valid_format", False)
                    }

                    final_results.append(combined_result)

                    logger.info(f"📋 Placa {i + 1}: '{char_results.get('plate_text', 'N/A')}' "
                                f"(Conf: {combined_result['overall_confidence']:.3f})")

                except Exception as e:
                    logger.error(f"❌ Error procesando placa {i + 1}: {str(e)}")
                    continue

            # Ordenar por confianza general
            final_results.sort(key=lambda x: x["overall_confidence"], reverse=True)

            # Resultado final
            pipeline_result = {
                "success": len(final_results) > 0,
                "plates_processed": len(final_results),
                "plate_detection": plate_results,
                "final_results": final_results,
                "best_result": final_results[0] if final_results else None,
                "processing_summary": {
                    "plates_detected": plate_results["plates_detected"],
                    "plates_with_text": len([r for r in final_results if r["plate_text"]]),
                    "valid_plates": len([r for r in final_results if r["is_valid_plate"]])
                }
            }

            logger.success(f"✅ Pipeline completado: {len(final_results)} placa(s) procesada(s)")

            return pipeline_result

        except Exception as e:
            logger.error(f"❌ Error en pipeline completo: {str(e)}")
            return {
                "success": False,
                "message": f"Error en pipeline: {str(e)}",
                "plate_detection": None,
                "character_recognition": None,
                "final_results": []
            }

    def _calculate_combined_confidence(self, plate_conf: float, char_conf: float) -> float:
        """Calcula la confianza combinada de detección + reconocimiento"""
        # Promedio ponderado: 40% detección, 60% reconocimiento
        return (plate_conf * 0.4) + (char_conf * 0.6)

    def get_best_plate_text(self, image_input, **kwargs) -> Optional[str]:
        """
        Obtiene el texto de la mejor placa detectada

        Returns:
            Texto de la placa con mayor confianza o None si no se detecta nada
        """
        try:
            result = self.process_full_pipeline(image_input, **kwargs)

            if result["success"] and result["best_result"]:
                return result["best_result"]["plate_text"]

            return None

        except Exception as e:
            logger.error(f"❌ Error obteniendo mejor placa: {str(e)}")
            return None

    def process_multiple_images(self, image_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Procesa múltiples imágenes en lote

        Args:
            image_paths: Lista de rutas de imágenes
            **kwargs: Parámetros adicionales

        Returns:
            Lista de resultados para cada imagen
        """
        results = []

        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"🖼️ Procesando imagen {i + 1}/{len(image_paths)}: {image_path}")

                result = self.process_full_pipeline(image_path, **kwargs)
                result["image_path"] = image_path
                result["image_index"] = i + 1

                results.append(result)

            except Exception as e:
                logger.error(f"❌ Error procesando {image_path}: {str(e)}")
                results.append({
                    "success": False,
                    "image_path": image_path,
                    "image_index": i + 1,
                    "error": str(e)
                })

        logger.info(f"📊 Lote completado: {len(results)} imágenes procesadas")
        return results


# Instancia global del gestor de modelos
model_manager = ModelManager()