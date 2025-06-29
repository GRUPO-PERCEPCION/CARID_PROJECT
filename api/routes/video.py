from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from typing import Optional
from loguru import logger

from api.dependencies import get_model_manager, log_request_info
from services.video_service import video_service
from services.file_service import file_service
from models.model_manager import ModelManager
from core.utils import is_valid_video, get_video_info
from config.settings import settings

# Router para endpoints de video
video_router = APIRouter(prefix="/api/v1/video", tags=["Video Processing"])

# ✅ OBTENER CONFIGURACIONES CENTRALIZADAS
video_config = settings.get_video_detection_config()
validation_config = settings.get_validation_config()

logger.info("🎬 Video routes inicializadas con configuración centralizada")
logger.debug(f"📊 Config video por defecto: {video_config}")


@video_router.post("/detect",
                   summary="Detectar Placas en Video",
                   description=f"""
                   Procesa un video para detectar y reconocer placas vehiculares con tracking inteligente y configuración centralizada.

                   **Características del procesamiento:**
                   - 🎯 Detección frame por frame optimizada
                   - 🔄 Tracking de placas para evitar duplicados
                   - 📊 Identificación de placas únicas por vehículo
                   - ✅ Validación de formato peruano
                   - 🏆 Selección de mejores detecciones por confianza

                   **Configuración por defecto (centralizada):**
                   - Confianza: {video_config['confidence_threshold']}
                   - IoU: {video_config['iou_threshold']}
                   - Frame skip: {video_config['frame_skip']}
                   - Duración máxima: {video_config['max_duration']}s
                   - Min frames detección: {video_config['min_detection_frames']}
                   - ROI habilitado: {settings.roi_enabled}
                   - Filtro 6 caracteres: {settings.force_six_characters}

                   **Formatos soportados:** MP4, AVI, MOV, MKV, WebM
                   **Duración máxima:** {video_config['max_duration']}s (configurable)
                   """)
async def detect_plates_in_video(
        file: UploadFile = File(..., description="Video a procesar"),
        confidence_threshold: Optional[float] = Form(None,
                                                     description=f"Umbral de confianza (por defecto: {video_config['confidence_threshold']})"),
        iou_threshold: Optional[float] = Form(None,
                                              description=f"Umbral IoU (por defecto: {video_config['iou_threshold']})"),
        frame_skip: Optional[int] = Form(None,
                                         description=f"Procesar cada N frames (por defecto: {video_config['frame_skip']})"),
        max_duration: Optional[int] = Form(None,
                                           description=f"Duración máxima en segundos (por defecto: {video_config['max_duration']})"),
        save_results: Optional[bool] = Form(None,
                                            description=f"Guardar resultados (por defecto: {video_config['save_results']})"),
        save_best_frames: Optional[bool] = Form(None,
                                                description=f"Guardar frames con mejores detecciones (por defecto: {video_config['save_best_frames']})"),
        create_annotated_video: Optional[bool] = Form(None,
                                                      description=f"Crear video anotado (por defecto: {video_config['create_annotated_video']})"),
        min_detection_frames: Optional[int] = Form(None,
                                                   description=f"Mínimo frames para confirmar placa (por defecto: {video_config['min_detection_frames']})"),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """✅ ACTUALIZADO: Endpoint principal para detección en videos usando configuración centralizada"""

    try:
        logger.info(f"🎬 Nueva solicitud de detección en video: {file.filename}")

        # Validar extensión de archivo
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nombre de archivo requerido"
            )

        # Verificar que sea un video válido por extensión (usando config centralizada)
        video_extensions = settings.video_extensions_list
        file_extension = file.filename.split('.')[-1].lower()

        if file_extension not in video_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Formato de video no soportado. Formatos válidos: {', '.join(video_extensions)}"
            )

        # ✅ CREAR PARÁMETROS CON FALLBACKS A CONFIG CENTRALIZADA
        request_params = {
            "confidence_threshold": confidence_threshold if confidence_threshold is not None else video_config[
                'confidence_threshold'],
            "iou_threshold": iou_threshold if iou_threshold is not None else video_config['iou_threshold'],
            "frame_skip": frame_skip if frame_skip is not None else video_config['frame_skip'],
            "max_duration": max_duration if max_duration is not None else video_config['max_duration'],
            "save_results": save_results if save_results is not None else video_config['save_results'],
            "save_best_frames": save_best_frames if save_best_frames is not None else video_config['save_best_frames'],
            "create_annotated_video": create_annotated_video if create_annotated_video is not None else video_config[
                'create_annotated_video'],
            "min_detection_frames": min_detection_frames if min_detection_frames is not None else video_config[
                'min_detection_frames']
        }

        # ✅ LOG DE CONFIGURACIÓN APLICADA
        logger.info(f"⚙️ Configuración aplicada: confidence={request_params['confidence_threshold']}, "
                    f"frame_skip={request_params['frame_skip']}, max_duration={request_params['max_duration']}")
        logger.debug(f"🔧 Parámetros completos: {request_params}")

        # ✅ VALIDAR PARÁMETROS USANDO CONFIG CENTRALIZADA
        validation = _validate_video_request_centralized(request_params)
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
        file_path, file_info = await file_service.save_upload_file(file, "video_")

        logger.info(f"💾 Video guardado: {file_info['filename']} ({file_info['size_mb']}MB)")

        # Verificar que sea un video válido
        if not is_valid_video(file_path):
            file_service.cleanup_temp_file(file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo no es un video válido o está corrupto"
            )

        # Obtener información básica del video
        basic_video_info = get_video_info(file_path)
        if not basic_video_info:
            file_service.cleanup_temp_file(file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudo leer la información del video"
            )

        # Verificar duración usando config centralizada
        max_duration_final = request_params['max_duration']
        if basic_video_info['duration_seconds'] > max_duration_final:
            file_service.cleanup_temp_file(file_path)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Video muy largo. Máximo: {max_duration_final}s, "
                       f"recibido: {basic_video_info['duration_seconds']:.1f}s"
            )

        logger.info(f"📹 Video válido: {basic_video_info['duration_seconds']:.1f}s, "
                    f"{basic_video_info['frame_count']} frames")

        # Procesar video con configuración centralizada
        result = await video_service.process_video(file_path, file_info, request_params)

        # Crear respuesta
        response = {
            "success": True,
            "message": "Procesamiento de video completado exitosamente con configuración centralizada",
            "data": result,
            "timestamp": result.get("timestamp"),
            "configuration_applied": {  # INFORMACIÓN DE CONFIG APLICADA
                "source": "centralized_settings",
                "video_config": video_config,
                "final_params": request_params,
                "roi_enabled": settings.roi_enabled,
                "force_six_characters": settings.force_six_characters,
                "tracking_config": settings.get_tracking_config()
            },
            # NUEVOS CAMPOS PARA MÚLTIPLES PLACAS EN VIDEO
            "video_analysis": {
                "total_unique_plates": len(result.get("unique_plates", [])),
                "spatial_regions_detected": len(
                    set(p.get("spatial_region", "R0_0") for p in result.get("unique_plates", []))),
                "plates_by_confidence": {
                    "high": [p for p in result.get("unique_plates", []) if p.get("best_confidence", 0) > 0.7],
                    "medium": [p for p in result.get("unique_plates", []) if 0.4 <= p.get("best_confidence", 0) <= 0.7],
                    "low": [p for p in result.get("unique_plates", []) if p.get("best_confidence", 0) < 0.4]
                },
                "plates_by_region": {},  # Se llenará con datos de regiones
                "tracking_effectiveness": {
                    "frames_processed": result.get("processing_summary", {}).get("frames_processed", 0),
                    "frames_with_detections": result.get("processing_summary", {}).get("frames_with_detections", 0),
                    "detection_rate": round(result.get("processing_summary", {}).get("frames_with_detections", 0) / max(
                        result.get("processing_summary", {}).get("frames_processed", 1), 1) * 100, 1)
                }
            },
            "all_unique_plates": result.get("unique_plates", [])
        }

        # Log del resultado
        if result["success"] and result["unique_plates"]:
            best_plate = result["best_plate"]
            logger.info(f"🎉 Video procesado: {len(result['unique_plates'])} placa(s) única(s). "
                        f"Mejor: '{best_plate['plate_text']}' "
                        f"(Confianza: {best_plate['best_confidence']:.3f})")
        else:
            logger.info("📭 No se detectaron placas válidas en el video")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en endpoint de detección de video: {str(e)}")
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


@video_router.post("/detect/quick",
                   summary="Detección Rápida en Video",
                   description=f"""
                   Versión optimizada para detección rápida en videos cortos usando configuración centralizada.

                   **Configuración rápida (centralizada):**
                   - Confianza: {settings.quick_confidence_threshold}
                   - Frame skip: {settings.quick_frame_skip}
                   - Duración máxima: {settings.quick_max_duration}s
                   - Sin guardar resultados para mayor velocidad
                   """)
async def quick_video_detect(
        file: UploadFile = File(...),
        confidence_threshold: Optional[float] = Form(None,
                                                     description=f"Umbral de confianza (por defecto: {settings.quick_confidence_threshold})"),
        frame_skip: Optional[int] = Form(None, description=f"Frame skip (por defecto: {settings.quick_frame_skip})"),
        max_duration: Optional[int] = Form(None,
                                           description=f"Duración máxima (por defecto: {settings.quick_max_duration})"),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """✅ ACTUALIZADO: Endpoint optimizado usando configuración centralizada"""

    try:
        # ✅ USAR CONFIGURACIÓN QUICK CENTRALIZADA
        quick_config = settings.get_quick_detection_config()

        # Parámetros optimizados para velocidad con fallbacks centralizados
        request_params = {
            "confidence_threshold": confidence_threshold if confidence_threshold is not None else quick_config[
                'confidence_threshold'],
            "iou_threshold": quick_config['iou_threshold'],
            "frame_skip": frame_skip if frame_skip is not None else quick_config['frame_skip'],
            "max_duration": max_duration if max_duration is not None else quick_config['max_duration'],
            "save_results": quick_config['save_results'],
            "save_best_frames": False,
            "create_annotated_video": False,
            "min_detection_frames": 1  # Menos restrictivo para velocidad
        }

        logger.info(f"⚡ Detección rápida de video con config centralizada: "
                    f"confidence={request_params['confidence_threshold']}, "
                    f"frame_skip={request_params['frame_skip']}")

        # Validar extensión
        if not file.filename or not any(file.filename.lower().endswith(ext)
                                        for ext in settings.video_extensions_list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Formato de video no soportado para detección rápida"
            )

        # Guardar archivo
        file_path, file_info = await file_service.save_upload_file(file, "quick_video_")

        # Verificar duración rápidamente usando config centralizada
        video_info = get_video_info(file_path)
        max_duration_final = request_params['max_duration']
        if video_info and video_info['duration_seconds'] > max_duration_final:
            file_service.cleanup_temp_file(file_path)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Video muy largo para detección rápida. Máximo: {max_duration_final}s"
            )

        # Procesar
        result = await video_service.process_video(file_path, file_info, request_params)

        # Respuesta simplificada
        if result["success"] and result["unique_plates"]:
            best_plate = result["best_plate"]
            return {
                "success": True,
                "unique_plates_count": len(result["unique_plates"]),
                "best_plate_text": best_plate["plate_text"],
                "best_confidence": best_plate["best_confidence"],
                "detection_count": best_plate["detection_count"],
                "is_valid_format": best_plate["is_valid_format"],
                "processing_time": result["processing_time"],
                "frames_processed": result["processing_summary"]["frames_processed"],
                "configuration": {  # ✅ INFO DE CONFIG USADA
                    "mode": "quick_video",
                    "source": "centralized_settings",
                    "params_used": request_params
                }
            }
        else:
            return {
                "success": False,
                "unique_plates_count": 0,
                "best_plate_text": "",
                "best_confidence": 0.0,
                "detection_count": 0,
                "is_valid_format": False,
                "processing_time": result["processing_time"],
                "frames_processed": result["processing_summary"]["frames_processed"],
                "message": "No se detectaron placas en el video",
                "configuration": {
                    "mode": "quick_video",
                    "source": "centralized_settings",
                    "params_used": request_params
                }
            }

    except Exception as e:
        logger.error(f"❌ Error en detección rápida de video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en detección rápida: {str(e)}"
        )


@video_router.get("/stats",
                  summary="Estadísticas del Servicio de Videos",
                  description="Información sobre el estado y rendimiento del procesamiento de videos con configuración centralizada")
async def get_video_stats(
        request_id: str = Depends(log_request_info)
):
    """✅ ACTUALIZADO: Obtiene estadísticas incluyendo configuración centralizada"""

    try:
        # Obtener estadísticas del sistema
        from api.dependencies import get_system_info
        system_info = get_system_info()

        # ✅ INFORMACIÓN ESPECÍFICA DE VIDEOS CON CONFIG CENTRALIZADA
        video_stats = {
            "centralized_configuration": {
                "video_detection": video_config,
                "tracking": settings.get_tracking_config(),
                "validation_ranges": validation_config,
                "quick_video": settings.get_quick_detection_config(),
                "roi_settings": settings.get_roi_config()
            },
            "configuration_summary": {
                "max_video_duration": video_config['max_duration'],
                "supported_formats": settings.video_extensions_list,
                "default_frame_skip": video_config['frame_skip'],
                "min_detection_frames": video_config['min_detection_frames'],
                "confidence_threshold": video_config['confidence_threshold'],
                "roi_enabled": settings.roi_enabled,
                "force_six_characters": settings.force_six_characters
            },
            "processing_capabilities": {
                "parallel_processing": True,
                "gpu_acceleration": system_info["gpu"]["cuda_available"],
                "max_concurrent_videos": 2,
                "roi_processing": settings.roi_enabled,
                "six_char_filtering": settings.force_six_characters,
                "auto_dash_formatting": True
            },
            "performance": {
                "avg_processing_speed": "2-5x tiempo real",
                "memory_usage": "Optimizado para videos largos",
                "threading": "AsyncIO + ThreadPoolExecutor",
                "config_source": "centralized_settings"
            }
        }

        return {
            "success": True,
            "message": "Estadísticas de video obtenidas exitosamente",
            "data": video_stats,
            "timestamp": __import__('time').time()
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas de video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo estadísticas: {str(e)}"
        )


@video_router.post("/validate-params",
                   summary="Validar Parámetros de Video",
                   description="Valida parámetros de procesamiento de video sin procesar archivos usando configuración centralizada")
async def validate_video_params(
        confidence_threshold: Optional[float] = Form(None,
                                                     description=f"Umbral de confianza (por defecto: {video_config['confidence_threshold']})"),
        iou_threshold: Optional[float] = Form(None,
                                              description=f"Umbral IoU (por defecto: {video_config['iou_threshold']})"),
        frame_skip: Optional[int] = Form(None, description=f"Frame skip (por defecto: {video_config['frame_skip']})"),
        max_duration: Optional[int] = Form(None,
                                           description=f"Duración máxima (por defecto: {video_config['max_duration']})"),
        min_detection_frames: Optional[int] = Form(None,
                                                   description=f"Min frames detección (por defecto: {video_config['min_detection_frames']})"),
        request_id: str = Depends(log_request_info)
):
    """✅ ACTUALIZADO: Valida parámetros usando configuración centralizada"""

    try:
        # ✅ USAR CONFIG CENTRALIZADA COMO FALLBACK
        params = {
            "confidence_threshold": confidence_threshold if confidence_threshold is not None else video_config[
                'confidence_threshold'],
            "iou_threshold": iou_threshold if iou_threshold is not None else video_config['iou_threshold'],
            "frame_skip": frame_skip if frame_skip is not None else video_config['frame_skip'],
            "max_duration": max_duration if max_duration is not None else video_config['max_duration'],
            "min_detection_frames": min_detection_frames if min_detection_frames is not None else video_config[
                'min_detection_frames']
        }

        logger.info(f"🔍 Validando parámetros de video con config centralizada: {params}")

        validation = _validate_video_request_centralized(params)

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
                    "frame_skip": frame_skip is None,
                    "max_duration": max_duration is None,
                    "min_detection_frames": min_detection_frames is None
                },
                "video_config": video_config,
                "validation_ranges": validation_config,
                "configuration_file": "settings.py + .env"
            }
        }

    except Exception as e:
        logger.error(f"❌ Error validando parámetros de video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validando parámetros: {str(e)}"
        )


def _validate_video_request_centralized(request_params: dict) -> dict:
    """✅ ACTUALIZADO: Valida los parámetros usando configuración centralizada"""
    validation = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": []
    }

    try:
        confidence = request_params.get('confidence_threshold', video_config['confidence_threshold'])
        iou = request_params.get('iou_threshold', video_config['iou_threshold'])
        frame_skip = request_params.get('frame_skip', video_config['frame_skip'])
        max_duration = request_params.get('max_duration', video_config['max_duration'])
        min_frames = request_params.get('min_detection_frames', video_config['min_detection_frames'])

        # ✅ USAR RANGOS DE VALIDACIÓN CENTRALIZADOS
        conf_range = validation_config['confidence_range']
        iou_range = validation_config['iou_range']
        frame_skip_range = validation_config['frame_skip_range']
        duration_range = validation_config['video_duration_range']
        warnings = validation_config['warnings']
        recommendations = validation_config['recommendations']

        # Validar umbrales usando config centralizada
        if confidence < conf_range[0] or confidence > conf_range[1]:
            validation["errors"].append(f"confidence_threshold debe estar entre {conf_range[0]} y {conf_range[1]}")

        if iou < iou_range[0] or iou > iou_range[1]:
            validation["errors"].append(f"iou_threshold debe estar entre {iou_range[0]} y {iou_range[1]}")

        if frame_skip < frame_skip_range[0] or frame_skip > frame_skip_range[1]:
            validation["errors"].append(f"frame_skip debe estar entre {frame_skip_range[0]} y {frame_skip_range[1]}")

        if max_duration < duration_range[0] or max_duration > duration_range[1]:
            validation["errors"].append(
                f"max_duration debe estar entre {duration_range[0]} y {duration_range[1]} segundos")

        if min_frames < 1 or min_frames > 10:
            validation["errors"].append("min_detection_frames debe estar entre 1 y 10")

        # ✅ ADVERTENCIAS USANDO CONFIG CENTRALIZADA
        if confidence < warnings['low_confidence']:
            validation["warnings"].append(
                f"confidence_threshold bajo ({confidence}) puede generar muchos falsos positivos en video")

        if frame_skip > warnings['high_frame_skip']:
            validation["warnings"].append(f"frame_skip alto ({frame_skip}) puede perder detecciones importantes")

        if max_duration > warnings['long_video']:
            validation["warnings"].append(f"Videos largos ({max_duration}s) requieren más tiempo de procesamiento")

        # ✅ RECOMENDACIONES USANDO CONFIG CENTRALIZADA
        rec_conf_range = recommendations['confidence_range']
        rec_frame_range = recommendations['frame_skip_range']

        if confidence > rec_conf_range[1]:
            validation["recommendations"].append(
                f"Para videos, considere usar confidence_threshold entre {rec_conf_range[0]}-{rec_conf_range[1]}")

        if frame_skip == 1:
            validation["recommendations"].append(
                f"frame_skip=1 es lento, considere usar {rec_frame_range[0]}-{rec_frame_range[1]} para mejor rendimiento")

        validation["is_valid"] = len(validation["errors"]) == 0

        # ✅ AGREGAR INFORMACIÓN DE CONFIGURACIÓN USADA
        validation["configuration_info"] = {
            "validation_source": "centralized_settings",
            "ranges_used": {
                "confidence": conf_range,
                "iou": iou_range,
                "frame_skip": frame_skip_range,
                "duration": duration_range
            },
            "warning_thresholds": warnings,
            "recommendation_ranges": recommendations,
            "video_config": video_config
        }

    except Exception as e:
        validation["is_valid"] = False
        validation["errors"].append(f"Error validando request: {str(e)}")

    return validation


# ✅ NUEVOS ENDPOINTS PARA CONFIGURACIÓN DINÁMICA

@video_router.get("/config",
                  summary="Obtener Configuración de Video Actual",
                  description="Obtiene la configuración actual del servicio de video")
async def get_video_config(
        request_id: str = Depends(log_request_info)
):
    """✅ NUEVO: Obtiene configuración actual de video"""
    try:
        current_config = video_service.get_current_config()

        return {
            "success": True,
            "message": "Configuración de video actual obtenida",
            "data": current_config,
            "metadata": {
                "source": "centralized_settings",
                "config_file": "settings.py + .env",
                "last_updated": __import__('time').time()
            }
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo configuración de video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo configuración: {str(e)}"
        )


@video_router.get("/config/recommendations/{video_type}",
                  summary="Obtener Recomendaciones para Tipo de Video",
                  description="Obtiene parámetros recomendados según el tipo de video")
async def get_video_config_recommendations(
        video_type: str,
        request_id: str = Depends(log_request_info)
):
    """✅ NUEVO: Obtiene recomendaciones por tipo de video"""
    try:
        valid_types = ["standard", "quick", "high_precision", "long_video"]

        if video_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de video inválido. Válidos: {valid_types}"
            )

        recommendations = video_service.get_recommended_params_for_video_type(video_type)

        return {
            "success": True,
            "video_type": video_type,
            "recommended_parameters": recommendations,
            "description": {
                "standard": "Configuración balanceada para videos normales",
                "quick": "Configuración optimizada para velocidad en videos cortos",
                "high_precision": "Configuración para máxima precisión (más lento)",
                "long_video": "Configuración optimizada para videos largos"
            }.get(video_type, "Configuración personalizada"),
            "source": "centralized_settings"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error obteniendo recomendaciones de video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo recomendaciones: {str(e)}"
        )


@video_router.post("/test-pipeline",
                   summary="Test Pipeline de Video",
                   description="Endpoint de testing para verificar el pipeline completo de video con configuración centralizada")
async def test_video_pipeline(
        file: UploadFile = File(...),
        debug: Optional[bool] = Form(False),
        video_type: Optional[str] = Form("standard", description="Tipo de configuración a usar"),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """✅ ACTUALIZADO: Endpoint de testing con configuración centralizada"""

    try:
        # ✅ OBTENER PARÁMETROS SEGÚN TIPO DE VIDEO
        test_config = video_service.get_recommended_params_for_video_type(video_type)

        # Parámetros de testing más permisivos
        request_params = {
            **test_config,
            "confidence_threshold": max(0.3, test_config['confidence_threshold'] - 0.1),  # Más permisivo
            "frame_skip": max(2, test_config.get('frame_skip', 3)),  # Procesar más frames
            "max_duration": min(120, test_config.get('max_duration', 600)),  # Máximo 2 minutos para testing
            "save_results": True,
            "save_best_frames": True,
            "create_annotated_video": False,
            "min_detection_frames": 1  # Permisivo para testing
        }

        logger.info(f"🧪 Test pipeline de video con tipo '{video_type}' y config centralizada")
        logger.debug(f"🔧 Parámetros de test: {request_params}")

        # Guardar archivo
        file_path, file_info = await file_service.save_upload_file(file, "test_video_")

        # Procesar con timing detallado
        import time
        start_time = time.time()

        result = await video_service.process_video(file_path, file_info, request_params)

        total_time = time.time() - start_time

        # Respuesta extendida para testing
        test_response = {
            "success": result["success"],
            "message": result["message"],
            "processing_times": {
                "total_seconds": round(total_time, 3),
                "reported_time": result["processing_time"]
            },
            "video_analysis": {
                "unique_plates_found": len(result["unique_plates"]),
                "best_plate": result["best_plate"]["plate_text"] if result["best_plate"] else None,
                "all_plates": [p["plate_text"] for p in result["unique_plates"]],
                "detection_counts": [p["detection_count"] for p in result["unique_plates"]],
                "confidence_scores": [p["best_confidence"] for p in result["unique_plates"]]
            },
            "processing_summary": result["processing_summary"],
            "file_info": result["file_info"],
            "video_info": result["video_info"],
            "urls": result["result_urls"],
            "configuration_test": {  # ✅ INFORMACIÓN DE CONFIG DE TEST
                "video_type_used": video_type,
                "test_parameters": request_params,
                "centralized_config": test_config,
                "source": "centralized_settings"
            }
        }

        if debug:
            # Información adicional para debug
            test_response["debug_info"] = {
                "frame_skip_used": request_params["frame_skip"],
                "confidence_threshold": request_params["confidence_threshold"],
                "model_status": models.get_model_info(),
                "centralized_settings": {
                    "roi_enabled": settings.roi_enabled,
                    "force_six_characters": settings.force_six_characters,
                    "roi_percentage": settings.roi_percentage,
                    "all_video_configs": video_service.get_current_config()
                },
                "tracking_details": {
                    "frames_processed": result["processing_summary"]["frames_processed"],
                    "frames_with_detections": result["processing_summary"]["frames_with_detections"],
                    "total_detections": result["processing_summary"]["total_detections"]
                }
            }

        return test_response

    except Exception as e:
        logger.error(f"❌ Error en test pipeline de video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en test pipeline: {str(e)}"
        )