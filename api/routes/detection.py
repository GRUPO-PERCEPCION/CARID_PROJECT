from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from typing import Optional
from loguru import logger

from api.dependencies import get_model_manager, log_request_info
from services.detection_service import detection_service
from services.file_service import file_service
from models.model_manager import ModelManager
from config.settings import settings

router = APIRouter(prefix="/api/v1/detect", tags=["Detection"])

# ✅ OBTENER CONFIGURACIONES CENTRALIZADAS
image_config = settings.get_image_detection_config()
validation_config = settings.get_validation_config()

logger.info("🔍 Detection routes inicializadas con configuración centralizada")
logger.debug(f"📊 Config por defecto: {image_config}")


@router.post("/image",
             summary="Detectar Placas en Imagen",
             description=f"""
             Procesa una imagen para detectar y reconocer placas vehiculares usando configuración centralizada.

             **Pipeline completo:**
             1. 🎯 Detección de regiones de placas
             2. 📖 Reconocimiento de caracteres  
             3. ✅ Validación de formato peruano
             4. 📊 Generación de resultados estructurados

             **Configuración por defecto (centralizada):**
             - Confianza: {image_config['confidence_threshold']}
             - IoU: {image_config['iou_threshold']}
             - Detecciones máximas: {image_config['max_detections']}
             - ROI habilitado: {settings.roi_enabled}
             - Filtro 6 caracteres: {settings.force_six_characters}

             **Formatos soportados:** JPG, JPEG, PNG
             **Tamaño máximo:** {settings.max_file_size}MB
             """)
async def detect_plates_in_image(
        file: UploadFile = File(..., description="Imagen a procesar"),
        confidence_threshold: Optional[float] = Form(None,
                                                     description=f"Umbral de confianza (por defecto: {image_config['confidence_threshold']})"),
        iou_threshold: Optional[float] = Form(None,
                                              description=f"Umbral IoU (por defecto: {image_config['iou_threshold']})"),
        max_detections: Optional[int] = Form(None,
                                             description=f"Máximo número de detecciones (por defecto: {image_config['max_detections']})"),
        enhance_image: Optional[bool] = Form(None,
                                             description=f"Aplicar mejoras a la imagen (por defecto: {image_config['enhance_image']})"),
        return_visualization: Optional[bool] = Form(None,
                                                    description=f"Retornar imagen con visualizaciones (por defecto: {image_config['return_visualization']})"),
        save_results: Optional[bool] = Form(None,
                                            description=f"Guardar resultados (por defecto: {image_config['save_results']})"),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """✅ ACTUALIZADO: Endpoint principal usando configuración centralizada"""

    try:
        logger.info(f"📥 Nueva solicitud de detección: {file.filename}")

        # ✅ CREAR PARÁMETROS CON FALLBACKS A CONFIG CENTRALIZADA
        request_params = {
            "confidence_threshold": confidence_threshold if confidence_threshold is not None else image_config[
                'confidence_threshold'],
            "iou_threshold": iou_threshold if iou_threshold is not None else image_config['iou_threshold'],
            "max_detections": max_detections if max_detections is not None else image_config['max_detections'],
            "enhance_image": enhance_image if enhance_image is not None else image_config['enhance_image'],
            "return_visualization": return_visualization if return_visualization is not None else image_config[
                'return_visualization'],
            "save_results": save_results if save_results is not None else image_config['save_results']
        }

        # ✅ LOG DE CONFIGURACIÓN APLICADA
        logger.info(f"⚙️ Configuración aplicada: confidence={request_params['confidence_threshold']}, "
                    f"iou={request_params['iou_threshold']}, max_det={request_params['max_detections']}")
        logger.debug(f"🔧 Parámetros completos: {request_params}")

        # Validar parámetros usando configuración centralizada
        validation = detection_service.validate_detection_request(request_params)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Parámetros inválidos",
                    "errors": validation["errors"],
                    "configuration_info": validation.get("configuration_info", {})
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
            "timestamp": result.get("timestamp"),
            "configuration_applied": {  # ✅ INFORMACIÓN DE CONFIG APLICADA
                "source": "centralized_settings",
                "confidence_threshold": request_params['confidence_threshold'],
                "iou_threshold": request_params['iou_threshold'],
                "max_detections": request_params['max_detections'],
                "roi_enabled": settings.roi_enabled,
                "force_six_characters": settings.force_six_characters
            },
            # NUEVOS CAMPOS PARA MÚLTIPLES DETECCIONES
            "detection_summary": {
                "total_plates_found": result.get("plates_processed", 0),
                "valid_plates_count": len(
                    [p for p in result.get("final_results", []) if p.get("is_valid_plate", False)]),
                "spatial_regions_covered": len(result.get("spatial_distribution", {})),
                "confidence_breakdown": result.get("plates_by_confidence", {}),
                "processing_method": "enhanced_multiple_detection"
            },
            "all_detected_plates": result.get("all_detected_plates", []),  # LISTA COMPLETA
            "spatial_analysis": {
                "regions_found": result.get("spatial_distribution", {}),
                "region_count": len(result.get("spatial_distribution", {}))
            }
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
                "error": str(e),
                "configuration_info": {
                    "using_centralized_config": True,
                    "config_source": "settings.py + .env"
                }
            }
        )


@router.post("/image/quick",
             summary="Detección Rápida",
             description=f"""
             Versión optimizada que retorna solo el texto de la mejor placa.

             **Configuración rápida (centralizada):**
             - Confianza: {settings.quick_confidence_threshold}
             - IoU: {settings.quick_iou_threshold}
             - Detecciones máximas: {settings.quick_max_detections}
             - Sin guardar resultados para mayor velocidad
             """)
async def quick_detect(
        file: UploadFile = File(...),
        confidence_threshold: Optional[float] = Form(None,
                                                     description=f"Umbral de confianza (por defecto: {settings.quick_confidence_threshold})"),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """✅ ACTUALIZADO: Endpoint optimizado usando configuración centralizada"""

    try:
        # ✅ USAR CONFIGURACIÓN QUICK CENTRALIZADA
        quick_config = settings.get_quick_detection_config()

        # Parámetros optimizados para velocidad con fallback centralizado
        request_params = {
            "confidence_threshold": confidence_threshold if confidence_threshold is not None else quick_config[
                'confidence_threshold'],
            "iou_threshold": quick_config['iou_threshold'],
            "max_detections": quick_config['max_detections'],
            "enhance_image": quick_config['enhance_image'],
            "return_visualization": quick_config['return_visualization'],
            "save_results": quick_config['save_results']
        }

        logger.info(f"⚡ Detección rápida con config centralizada: confidence={request_params['confidence_threshold']}")

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
                "processing_time": result["processing_time"],
                "configuration": {  # ✅ INFO DE CONFIG USADA
                    "mode": "quick",
                    "source": "centralized_settings",
                    "confidence_used": request_params['confidence_threshold']
                }
            }
        else:
            return {
                "success": False,
                "plate_text": "",
                "confidence": 0.0,
                "is_valid_format": False,
                "processing_time": result["processing_time"],
                "message": "No se detectaron placas",
                "configuration": {
                    "mode": "quick",
                    "source": "centralized_settings",
                    "confidence_used": request_params['confidence_threshold']
                }
            }

    except Exception as e:
        logger.error(f"❌ Error en detección rápida: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en detección rápida: {str(e)}"
        )


@router.get("/stats",
            summary="Estadísticas del Servicio",
            description="Información sobre el estado y rendimiento del servicio de detección con configuración centralizada")
async def get_detection_stats(
        request_id: str = Depends(log_request_info)
):
    """✅ ACTUALIZADO: Obtiene estadísticas incluyendo configuración centralizada"""

    try:
        stats = await detection_service.get_processing_stats()

        # ✅ AGREGAR INFORMACIÓN DE CONFIGURACIÓN CENTRALIZADA
        enhanced_stats = {
            **stats,
            "centralized_configuration": {
                "image_detection": image_config,
                "validation_ranges": validation_config,
                "quick_detection": settings.get_quick_detection_config(),
                "roi_settings": settings.get_roi_config(),
                "plate_detector": settings.get_plate_detector_config(),
                "char_recognizer": settings.get_char_recognizer_config()
            },
            "configuration_source": "settings.py + .env file",
            "dynamic_config_available": True
        }

        return {
            "success": True,
            "message": "Estadísticas obtenidas exitosamente",
            "data": enhanced_stats,
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
             description="Valida parámetros de detección sin procesar imágenes usando configuración centralizada")
async def validate_detection_params(
        confidence_threshold: Optional[float] = Form(None,
                                                     description=f"Umbral de confianza (por defecto: {image_config['confidence_threshold']})"),
        iou_threshold: Optional[float] = Form(None,
                                              description=f"Umbral IoU (por defecto: {image_config['iou_threshold']})"),
        max_detections: Optional[int] = Form(None,
                                             description=f"Máximo detecciones (por defecto: {image_config['max_detections']})"),
        request_id: str = Depends(log_request_info)
):
    """✅ ACTUALIZADO: Valida parámetros usando configuración centralizada"""

    try:
        # ✅ USAR CONFIG CENTRALIZADA COMO FALLBACK
        params = {
            "confidence_threshold": confidence_threshold if confidence_threshold is not None else image_config[
                'confidence_threshold'],
            "iou_threshold": iou_threshold if iou_threshold is not None else image_config['iou_threshold'],
            "max_detections": max_detections if max_detections is not None else image_config['max_detections']
        }

        logger.info(f"🔍 Validando parámetros con config centralizada: {params}")

        validation = detection_service.validate_detection_request(params)

        return {
            "is_valid": validation["is_valid"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "recommendations": validation.get("recommendations", []),
            "parameters": params,
            "configuration_info": {  # ✅ INFORMACIÓN ADICIONAL
                "validation_source": "centralized_settings",
                "default_values_used": {
                    "confidence_threshold": confidence_threshold is None,
                    "iou_threshold": iou_threshold is None,
                    "max_detections": max_detections is None
                },
                "validation_ranges": validation_config,
                "configuration_file": "settings.py + .env"
            }
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


# ✅ NUEVOS ENDPOINTS PARA CONFIGURACIÓN DINÁMICA

@router.get("/config",
            summary="Obtener Configuración Actual",
            description="Obtiene la configuración actual del servicio de detección")
async def get_current_config(
        request_id: str = Depends(log_request_info)
):
    """✅ NUEVO: Obtiene configuración actual"""
    try:
        current_config = detection_service.get_current_config()

        return {
            "success": True,
            "message": "Configuración actual obtenida",
            "data": current_config,
            "metadata": {
                "source": "centralized_settings",
                "config_file": "settings.py + .env",
                "last_updated": __import__('time').time()
            }
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo configuración: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo configuración: {str(e)}"
        )


@router.get("/config/recommendations/{context}",
            summary="Obtener Recomendaciones de Configuración",
            description="Obtiene parámetros recomendados según el contexto")
async def get_config_recommendations(
        context: str,
        request_id: str = Depends(log_request_info)
):
    """✅ NUEVO: Obtiene recomendaciones por contexto"""
    try:
        valid_contexts = ["standard", "quick", "high_precision", "high_recall"]

        if context not in valid_contexts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Contexto inválido. Válidos: {valid_contexts}"
            )

        recommendations = detection_service.get_recommended_params_for_context(context)

        return {
            "success": True,
            "context": context,
            "recommended_parameters": recommendations,
            "description": {
                "standard": "Configuración balanceada para uso general",
                "quick": "Configuración optimizada para velocidad",
                "high_precision": "Configuración para máxima precisión",
                "high_recall": "Configuración para capturar más detecciones"
            }.get(context, "Configuración personalizada"),
            "source": "centralized_settings"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error obteniendo recomendaciones: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo recomendaciones: {str(e)}"
        )


@router.post("/test-pipeline",
             summary="Test Pipeline",
             description="Endpoint de testing para verificar el pipeline completo con configuración centralizada")
async def test_pipeline(
        file: UploadFile = File(...),
        debug: Optional[bool] = Form(False),
        context: Optional[str] = Form("standard", description="Contexto de configuración a usar"),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """✅ ACTUALIZADO: Endpoint de testing con configuración centralizada"""

    try:
        # ✅ OBTENER PARÁMETROS SEGÚN CONTEXTO
        if context == "quick":
            test_config = settings.get_quick_detection_config()
        else:
            test_config = detection_service.get_recommended_params_for_context(context)

        # Parámetros de testing más permisivos
        request_params = {
            **test_config,
            "confidence_threshold": max(0.3, test_config['confidence_threshold'] - 0.2),  # Más permisivo
            "max_detections": min(10, test_config.get('max_detections', 5) * 2),  # Más detecciones
            "return_visualization": True,
            "save_results": True
        }

        logger.info(f"🧪 Test pipeline con contexto '{context}' y config centralizada")
        logger.debug(f"🔧 Parámetros de test: {request_params}")

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
            "urls": result["result_urls"],
            "configuration_test": {  # ✅ INFORMACIÓN DE CONFIG DE TEST
                "context_used": context,
                "test_parameters": request_params,
                "centralized_config": test_config,
                "source": "centralized_settings"
            }
        }

        if debug:
            # Información adicional para debug
            test_response["debug_info"] = {
                "plate_detection": result["plate_detection"],
                "processing_summary": result["processing_summary"],
                "model_status": models.get_model_info(),
                "centralized_settings": {
                    "roi_enabled": settings.roi_enabled,
                    "force_six_characters": settings.force_six_characters,
                    "roi_percentage": settings.roi_percentage,
                    "all_configs": detection_service.get_current_config()
                }
            }

        return test_response

    except Exception as e:
        logger.error(f"❌ Error en test pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en test pipeline: {str(e)}"
        )