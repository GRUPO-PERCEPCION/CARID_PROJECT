from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from typing import Optional
from loguru import logger

from api.dependencies import get_model_manager, log_request_info
from services.detection_service import detection_service
from services.file_service import file_service
from models.model_manager import ModelManager

router = APIRouter(prefix="/api/v1/detect", tags=["Detection"])


@router.post("/image",
             summary="Detectar Placas en Imagen",
             description="""
             Procesa una imagen para detectar y reconocer placas vehiculares.

             **Pipeline completo:**
             1. 🎯 Detección de regiones de placas
             2. 📖 Reconocimiento de caracteres  
             3. ✅ Validación de formato peruano
             4. 📊 Generación de resultados estructurados

             **Formatos soportados:** JPG, JPEG, PNG
             **Tamaño máximo:** 50MB
             """)
async def detect_plates_in_image(
        file: UploadFile = File(..., description="Imagen a procesar"),
        confidence_threshold: Optional[float] = Form(0.5, description="Umbral de confianza (0.1-1.0)"),
        iou_threshold: Optional[float] = Form(0.4, description="Umbral IoU (0.1-1.0)"),
        max_detections: Optional[int] = Form(5, description="Máximo número de detecciones (1-10)"),
        enhance_image: Optional[bool] = Form(False, description="Aplicar mejoras a la imagen"),
        return_visualization: Optional[bool] = Form(False, description="Retornar imagen con visualizaciones"),
        save_results: Optional[bool] = Form(True, description="Guardar resultados"),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """Endpoint principal para detección en imágenes"""

    try:
        logger.info(f"📥 Nueva solicitud de detección: {file.filename}")

        # Crear parámetros de solicitud
        request_params = {
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "max_detections": max_detections,
            "enhance_image": enhance_image,
            "return_visualization": return_visualization,
            "save_results": save_results
        }

        # Validar parámetros
        validation = detection_service.validate_detection_request(request_params)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Parámetros inválidos",
                    "errors": validation["errors"]
                }
            )

        # Mostrar advertencias si las hay
        if validation["warnings"]:
            logger.warning(f"⚠️ Advertencias: {validation['warnings']}")

        # Guardar archivo subido
        file_path, file_info = await file_service.save_upload_file(file, "detection_")

        logger.info(f"💾 Archivo guardado: {file_info['filename']} ({file_info['size_mb']}MB)")

        # Procesar imagen
        result = await detection_service.process_image(file_path, file_info, request_params)

        # Crear respuesta
        response = {
            "success": True,
            "message": "Procesamiento completado exitosamente",
            "data": result,
            "timestamp": result.get("timestamp")
        }

        # Log del resultado
        if result["success"] and result["best_result"]:
            logger.info(f"🎉 Mejor resultado: '{result['best_result']['plate_text']}' "
                        f"(Confianza: {result['best_result']['overall_confidence']:.3f})")
        else:
            logger.info("📭 No se detectaron placas válidas")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en endpoint de detección: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Error interno del servidor",
                "error": str(e)
            }
        )


@router.post("/image/quick",
             summary="Detección Rápida",
             description="Versión optimizada que retorna solo el texto de la mejor placa")
async def quick_detect(
        file: UploadFile = File(...),
        confidence_threshold: Optional[float] = Form(0.6),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """Endpoint optimizado para detección rápida"""

    try:
        # Parámetros optimizados para velocidad
        request_params = {
            "confidence_threshold": confidence_threshold,
            "iou_threshold": 0.4,
            "max_detections": 3,  # Menos detecciones = más rápido
            "enhance_image": False,
            "return_visualization": False,
            "save_results": False  # No guardar para mayor velocidad
        }

        # Guardar archivo
        file_path, file_info = await file_service.save_upload_file(file, "quick_")

        # Procesar
        result = await detection_service.process_image(file_path, file_info, request_params)

        # Respuesta simplificada
        if result["success"] and result["best_result"]:
            return {
                "success": True,
                "plate_text": result["best_result"]["plate_text"],
                "confidence": result["best_result"]["overall_confidence"],
                "is_valid_format": result["best_result"]["is_valid_plate"],
                "processing_time": result["processing_time"]
            }
        else:
            return {
                "success": False,
                "plate_text": "",
                "confidence": 0.0,
                "is_valid_format": False,
                "processing_time": result["processing_time"],
                "message": "No se detectaron placas"
            }

    except Exception as e:
        logger.error(f"❌ Error en detección rápida: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en detección rápida: {str(e)}"
        )


@router.get("/stats",
            summary="Estadísticas del Servicio",
            description="Información sobre el estado y rendimiento del servicio de detección")
async def get_detection_stats(
        request_id: str = Depends(log_request_info)
):
    """Obtiene estadísticas del servicio de detección"""

    try:
        stats = await detection_service.get_processing_stats()

        return {
            "success": True,
            "message": "Estadísticas obtenidas exitosamente",
            "data": stats,
            "timestamp": __import__('time').time()
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo estadísticas: {str(e)}"
        )


@router.post("/validate-params",
             summary="Validar Parámetros",
             description="Valida parámetros de detección sin procesar imágenes")
async def validate_detection_params(
        confidence_threshold: Optional[float] = Form(0.5),
        iou_threshold: Optional[float] = Form(0.4),
        max_detections: Optional[int] = Form(5),
        request_id: str = Depends(log_request_info)
):
    """Valida parámetros de detección"""

    try:
        params = {
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "max_detections": max_detections
        }

        validation = detection_service.validate_detection_request(params)

        return {
            "is_valid": validation["is_valid"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "parameters": params
        }

    except Exception as e:
        logger.error(f"❌ Error validando parámetros: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validando parámetros: {str(e)}"
        )


@router.delete("/cleanup",
               summary="Limpiar Archivos Temporales",
               description="Elimina archivos temporales antiguos")
async def cleanup_temp_files(
        max_age_hours: Optional[int] = 24,
        request_id: str = Depends(log_request_info)
):
    """Limpia archivos temporales"""

    try:
        # Ejecutar limpieza
        file_service.cleanup_old_files(max_age_hours)

        return {
            "success": True,
            "message": f"Limpieza completada: archivos más antiguos que {max_age_hours} horas eliminados",
            "timestamp": __import__('time').time()
        }

    except Exception as e:
        logger.error(f"❌ Error en limpieza: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en limpieza: {str(e)}"
        )


# Endpoint para testing y desarrollo
@router.post("/test-pipeline",
             summary="Test Pipeline",
             description="Endpoint de testing para verificar el pipeline completo")
async def test_pipeline(
        file: UploadFile = File(...),
        debug: Optional[bool] = Form(False),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """Endpoint de testing con información detallada"""

    try:
        # Parámetros de testing
        request_params = {
            "confidence_threshold": 0.3,  # Bajo para capturar más detecciones
            "iou_threshold": 0.4,
            "max_detections": 10,
            "enhance_image": True,
            "return_visualization": True,
            "save_results": True
        }

        # Guardar archivo
        file_path, file_info = await file_service.save_upload_file(file, "test_")

        # Procesar con timing detallado
        import time
        start_time = time.time()

        result = await detection_service.process_image(file_path, file_info, request_params)

        total_time = time.time() - start_time

        # Respuesta extendida para testing
        test_response = {
            "success": result["success"],
            "message": result["message"],
            "processing_times": {
                "total_seconds": round(total_time, 3),
                "reported_time": result["processing_time"]
            },
            "detection_details": {
                "plates_found": result["plates_processed"],
                "best_plate": result["best_result"]["plate_text"] if result["best_result"] else None,
                "all_plates": [p["plate_text"] for p in result["final_results"]],
                "confidences": [p["overall_confidence"] for p in result["final_results"]]
            },
            "file_info": result["file_info"],
            "urls": result["result_urls"]
        }

        if debug:
            # Información adicional para debug
            test_response["debug_info"] = {
                "plate_detection": result["plate_detection"],
                "processing_summary": result["processing_summary"],
                "model_status": models.get_model_info()
            }

        return test_response

    except Exception as e:
        logger.error(f"❌ Error en test pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en test pipeline: {str(e)}"
        )