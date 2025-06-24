import time
from typing import Dict, Any, Optional, List
import numpy as np
from loguru import logger

from models.model_manager import model_manager
from services.file_service import file_service
from core.utils import PerformanceTimer


class DetectionService:
    """Servicio principal para detección y reconocimiento de placas"""

    def __init__(self):
        self.model_manager = model_manager
        self.file_service = file_service

    async def process_image(
            self,
            file_path: str,
            file_info: Dict[str, Any],
            request_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa una imagen completa con el pipeline de detección + reconocimiento

        Args:
            file_path: Ruta de la imagen a procesar
            file_info: Información del archivo
            request_params: Parámetros de la solicitud

        Returns:
            Dict con todos los resultados
        """
        start_time = time.time()
        result_id = self.file_service.create_result_id()

        try:
            logger.info(f"🔄 Iniciando procesamiento de imagen: {file_info['filename']}")

            # Verificar que los modelos estén cargados
            if not self.model_manager.is_loaded:
                raise Exception("Los modelos no están cargados")

            # Parámetros para los modelos
            model_kwargs = {
                'conf': request_params.get('confidence_threshold', 0.5),
                'iou': request_params.get('iou_threshold', 0.4),
                'verbose': False
            }

            # Procesar con el pipeline completo
            with PerformanceTimer("Pipeline completo"):
                pipeline_result = self.model_manager.process_full_pipeline(
                    file_path,
                    **model_kwargs
                )

            # Guardar imágenes si se solicita
            result_urls = {}
            if request_params.get('save_results', True) and pipeline_result.get("success"):
                try:
                    # Copiar imagen original
                    original_path = await self.file_service.copy_to_results(
                        file_path, result_id, "original"
                    )
                    result_urls["original"] = self.file_service.get_file_url(original_path)

                    # Generar visualización si se solicita
                    if request_params.get('return_visualization', False):
                        visualization = await self._create_visualization(
                            file_path, pipeline_result, result_id
                        )
                        if visualization:
                            result_urls["visualization"] = visualization

                except Exception as e:
                    logger.warning(f"⚠️ Error guardando resultados: {str(e)}")

            # Tiempo de procesamiento
            processing_time = time.time() - start_time

            # Crear resultado final con estructura compatible
            processed_plates = []
            if pipeline_result.get("success") and pipeline_result.get("final_results"):
                for result in pipeline_result["final_results"]:
                    processed_plate = {
                        "plate_id": result["plate_id"],
                        "plate_bbox": result["plate_bbox"],
                        "plate_confidence": result["plate_confidence"],
                        "plate_area": result["plate_area"],
                        "character_recognition": result["character_recognition"],
                        "plate_text": result["plate_text"],
                        "overall_confidence": result["overall_confidence"],
                        "is_valid_plate": result["is_valid_plate"]
                    }
                    processed_plates.append(processed_plate)

            # Crear resultado final
            result = {
                "success": pipeline_result.get("success", False),
                "message": self._generate_result_message(pipeline_result, processed_plates),
                "file_info": file_info,
                "plates_processed": len(processed_plates),
                "plate_detection": pipeline_result.get("plate_detection"),
                "final_results": processed_plates,
                "best_result": processed_plates[0] if processed_plates else None,
                "processing_summary": {
                    "plates_detected": pipeline_result.get("plates_processed", 0),
                    "plates_with_text": len([p for p in processed_plates if p["plate_text"]]),
                    "valid_plates": len([p for p in processed_plates if p["is_valid_plate"]])
                },
                "processing_time": round(processing_time, 3),
                "timestamp": time.time(),
                "result_urls": result_urls if result_urls else None
            }

            # Limpiar archivo temporal
            self.file_service.cleanup_temp_file(file_path)

            logger.success(f"✅ Procesamiento completado en {processing_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"❌ Error procesando imagen: {str(e)}")

            # Limpiar archivo temporal en caso de error
            self.file_service.cleanup_temp_file(file_path)

            processing_time = time.time() - start_time

            # Retornar resultado de error
            return {
                "success": False,
                "message": f"Error procesando imagen: {str(e)}",
                "file_info": file_info,
                "plates_processed": 0,
                "plate_detection": None,
                "final_results": [],
                "best_result": None,
                "processing_summary": {
                    "plates_detected": 0,
                    "plates_with_text": 0,
                    "valid_plates": 0
                },
                "processing_time": round(processing_time, 3),
                "timestamp": time.time(),
                "result_urls": None
            }

    def _generate_result_message(self, pipeline_result: Dict[str, Any], processed_plates: List[Dict]) -> str:
        """Genera mensaje descriptivo del resultado"""
        if not pipeline_result.get("success"):
            return "No se detectaron placas en la imagen"

        if not processed_plates:
            return "Se detectaron regiones de placas pero no se pudo reconocer texto"

        valid_plates = [p for p in processed_plates if p["is_valid_plate"]]

        if valid_plates:
            best_plate = valid_plates[0]
            return f"Se detectó placa válida: '{best_plate['plate_text']}' con confianza {best_plate['overall_confidence']:.2f}"
        else:
            best_plate = processed_plates[0]
            return f"Se detectó texto: '{best_plate['plate_text']}' pero formato no válido"

    async def _create_visualization(self, file_path: str, pipeline_result: Dict[str, Any], result_id: str) -> Optional[
        str]:
        """
        Crea una imagen de visualización con las detecciones

        Returns:
            URL de la imagen de visualización o None si hay error
        """
        try:
            # Crear visualización de detección de placas
            if pipeline_result.get("plate_detection") and pipeline_result["plate_detection"]["success"]:
                plate_viz = self.model_manager.plate_detector.visualize_detections(file_path)

                # Guardar visualización
                viz_path = await self.file_service.save_result_image(
                    plate_viz, result_id, "visualization"
                )

                return self.file_service.get_file_url(viz_path)

            return None

        except Exception as e:
            logger.warning(f"⚠️ Error creando visualización: {str(e)}")
            return None

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del servicio de procesamiento"""
        try:
            # Información de modelos
            model_info = self.model_manager.get_model_info()

            # Información de archivos
            temp_files = list(self.file_service.temp_dir.glob("*"))
            result_files = list(self.file_service.results_dir.glob("*"))

            return {
                "models_status": {
                    "loaded": model_info["models_loaded"],
                    "device": model_info["device"],
                    "plate_detector": model_info["plate_detector_loaded"],
                    "char_recognizer": model_info["char_recognizer_loaded"]
                },
                "file_system": {
                    "temp_files_count": len(temp_files),
                    "result_files_count": len(result_files),
                    "temp_dir_size_mb": sum(f.stat().st_size for f in temp_files if f.is_file()) / (1024 * 1024),
                    "results_dir_size_mb": sum(f.stat().st_size for f in result_files if f.is_file()) / (1024 * 1024)
                },
                "configuration": {
                    "max_file_size_mb": 50,
                    "confidence_threshold": model_info["confidence_threshold"],
                    "iou_threshold": model_info["iou_threshold"]
                }
            }

        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {str(e)}")
            return {"error": str(e)}

    def validate_detection_request(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Valida los parámetros de la solicitud de detección"""
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }

        try:
            confidence = request_params.get('confidence_threshold', 0.5)
            iou = request_params.get('iou_threshold', 0.4)
            max_det = request_params.get('max_detections', 5)

            # Validar umbrales
            if confidence < 0.1 or confidence > 1.0:
                validation["errors"].append("confidence_threshold debe estar entre 0.1 y 1.0")

            if iou < 0.1 or iou > 1.0:
                validation["errors"].append("iou_threshold debe estar entre 0.1 y 1.0")

            if max_det < 1 or max_det > 10:
                validation["errors"].append("max_detections debe estar entre 1 y 10")

            # Advertencias
            if confidence < 0.3:
                validation["warnings"].append("confidence_threshold muy bajo, puede generar muchos falsos positivos")

            if confidence > 0.8:
                validation["warnings"].append("confidence_threshold muy alto, puede perderse detecciones válidas")

            validation["is_valid"] = len(validation["errors"]) == 0

        except Exception as e:
            validation["is_valid"] = False
            validation["errors"].append(f"Error validando request: {str(e)}")

        return validation


# Instancia global del servicio
detection_service = DetectionService()