"""
Rutas para procesamiento de videos con reconocimiento de placas
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from typing import Optional
from loguru import logger

from api.dependencies import get_model_manager, log_request_info
from services.video_service import video_service
from services.file_service import file_service
from models.model_manager import ModelManager
from core.utils import is_valid_video, get_video_info

# Router para endpoints de video
video_router = APIRouter(prefix="/api/v1/video", tags=["Video Processing"])


@video_router.post("/detect",
                   summary="Detectar Placas en Video",
                   description="""
                   Procesa un video para detectar y reconocer placas vehiculares con tracking inteligente.

                   **Características del procesamiento:**
                   - 🎯 Detección frame por frame optimizada
                   - 🔄 Tracking de placas para evitar duplicados
                   - 📊 Identificación de placas únicas por vehículo
                   - ✅ Validación de formato peruano
                   - 🏆 Selección de mejores detecciones por confianza

                   **Formatos soportados:** MP4, AVI, MOV, MKV, WebM
                   **Duración máxima:** 5 minutos (configurable)
                   """)
async def detect_plates_in_video(
        file: UploadFile = File(..., description="Video a procesar"),
        confidence_threshold: Optional[float] = Form(0.4, description="Umbral de confianza (0.1-1.0)"),
        iou_threshold: Optional[float] = Form(0.4, description="Umbral IoU (0.1-1.0)"),
        frame_skip: Optional[int] = Form(3, description="Procesar cada N frames (1-10)"),
        max_duration: Optional[int] = Form(300, description="Duración máxima en segundos"),
        save_results: Optional[bool] = Form(True, description="Guardar resultados"),
        save_best_frames: Optional[bool] = Form(True, description="Guardar frames con mejores detecciones"),
        create_annotated_video: Optional[bool] = Form(False, description="Crear video anotado"),
        min_detection_frames: Optional[int] = Form(2, description="Mínimo frames para confirmar placa"),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """Endpoint principal para detección en videos"""

    try:
        logger.info(f"🎬 Nueva solicitud de detección en video: {file.filename}")

        # Validar extensión de archivo
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nombre de archivo requerido"
            )

        # Verificar que sea un video válido por extensión
        video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        file_extension = file.filename.split('.')[-1].lower()

        if file_extension not in video_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Formato de video no soportado. Formatos válidos: {', '.join(video_extensions)}"
            )

        # Crear parámetros de solicitud
        request_params = {
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "frame_skip": frame_skip,
            "max_duration": max_duration,
            "save_results": save_results,
            "save_best_frames": save_best_frames,
            "create_annotated_video": create_annotated_video,
            "min_detection_frames": min_detection_frames
        }

        # Validar parámetros
        validation = _validate_video_request(request_params)
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

        # Verificar duración
        if basic_video_info['duration_seconds'] > max_duration:
            file_service.cleanup_temp_file(file_path)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Video muy largo. Máximo: {max_duration}s, "
                       f"recibido: {basic_video_info['duration_seconds']:.1f}s"
            )

        logger.info(f"📹 Video válido: {basic_video_info['duration_seconds']:.1f}s, "
                    f"{basic_video_info['frame_count']} frames")

        # Procesar video
        result = await video_service.process_video(file_path, file_info, request_params)

        # Crear respuesta
        response = {
            "success": True,
            "message": "Procesamiento de video completado exitosamente",
            "data": result,
            "timestamp": result.get("timestamp")
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
                "error": str(e)
            }
        )


@video_router.post("/detect/quick",
                   summary="Detección Rápida en Video",
                   description="Versión optimizada para detección rápida en videos cortos")
async def quick_video_detect(
        file: UploadFile = File(...),
        confidence_threshold: Optional[float] = Form(0.5),
        frame_skip: Optional[int] = Form(5),  # Más agresivo para velocidad
        max_duration: Optional[int] = Form(60),  # Máximo 1 minuto
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """Endpoint optimizado para detección rápida en videos"""

    try:
        # Parámetros optimizados para velocidad
        request_params = {
            "confidence_threshold": confidence_threshold,
            "iou_threshold": 0.4,
            "frame_skip": frame_skip,
            "max_duration": max_duration,
            "save_results": False,  # No guardar para mayor velocidad
            "save_best_frames": False,
            "create_annotated_video": False,
            "min_detection_frames": 1  # Menos restrictivo
        }

        # Validar extensión
        if not file.filename or not any(file.filename.lower().endswith(ext)
                                        for ext in ['.mp4', '.avi', '.mov']):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Formato de video no soportado para detección rápida"
            )

        # Guardar archivo
        file_path, file_info = await file_service.save_upload_file(file, "quick_video_")

        # Verificar duración rápidamente
        video_info = get_video_info(file_path)
        if video_info and video_info['duration_seconds'] > max_duration:
            file_service.cleanup_temp_file(file_path)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Video muy largo para detección rápida. Máximo: {max_duration}s"
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
                "frames_processed": result["processing_summary"]["frames_processed"]
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
                "message": "No se detectaron placas en el video"
            }

    except Exception as e:
        logger.error(f"❌ Error en detección rápida de video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en detección rápida: {str(e)}"
        )


@video_router.get("/stats",
                  summary="Estadísticas del Servicio de Videos",
                  description="Información sobre el estado y rendimiento del procesamiento de videos")
async def get_video_stats(
        request_id: str = Depends(log_request_info)
):
    """Obtiene estadísticas del servicio de videos"""

    try:
        # Obtener estadísticas del sistema
        from api.dependencies import get_system_info
        system_info = get_system_info()

        # Información específica de videos
        video_stats = {
            "configuration": {
                "max_video_duration": 300,
                "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"],
                "default_frame_skip": 3,
                "min_detection_frames": 2
            },
            "processing_capabilities": {
                "parallel_processing": True,
                "gpu_acceleration": system_info["gpu"]["cuda_available"],
                "max_concurrent_videos": 2
            },
            "performance": {
                "avg_processing_speed": "2-5x tiempo real",
                "memory_usage": "Optimizado para videos largos",
                "threading": "AsyncIO + ThreadPoolExecutor"
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
                   description="Valida parámetros de procesamiento de video sin procesar archivos")
async def validate_video_params(
        confidence_threshold: Optional[float] = Form(0.4),
        iou_threshold: Optional[float] = Form(0.4),
        frame_skip: Optional[int] = Form(3),
        max_duration: Optional[int] = Form(300),
        min_detection_frames: Optional[int] = Form(2),
        request_id: str = Depends(log_request_info)
):
    """Valida parámetros de procesamiento de video"""

    try:
        params = {
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "frame_skip": frame_skip,
            "max_duration": max_duration,
            "min_detection_frames": min_detection_frames
        }

        validation = _validate_video_request(params)

        return {
            "is_valid": validation["is_valid"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "parameters": params,
            "recommendations": validation.get("recommendations", [])
        }

    except Exception as e:
        logger.error(f"❌ Error validando parámetros de video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validando parámetros: {str(e)}"
        )


def _validate_video_request(request_params: dict) -> dict:
    """Valida los parámetros de la solicitud de video"""
    validation = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": []
    }

    try:
        confidence = request_params.get('confidence_threshold', 0.4)
        iou = request_params.get('iou_threshold', 0.4)
        frame_skip = request_params.get('frame_skip', 3)
        max_duration = request_params.get('max_duration', 300)
        min_frames = request_params.get('min_detection_frames', 2)

        # Validar umbrales
        if confidence < 0.1 or confidence > 1.0:
            validation["errors"].append("confidence_threshold debe estar entre 0.1 y 1.0")

        if iou < 0.1 or iou > 1.0:
            validation["errors"].append("iou_threshold debe estar entre 0.1 y 1.0")

        if frame_skip < 1 or frame_skip > 10:
            validation["errors"].append("frame_skip debe estar entre 1 y 10")

        if max_duration < 10 or max_duration > 600:
            validation["errors"].append("max_duration debe estar entre 10 y 600 segundos")

        if min_frames < 1 or min_frames > 10:
            validation["errors"].append("min_detection_frames debe estar entre 1 y 10")

        # Advertencias para videos
        if confidence < 0.3:
            validation["warnings"].append("confidence_threshold bajo puede generar muchos falsos positivos en video")

        if frame_skip > 5:
            validation["warnings"].append("frame_skip alto puede perder detecciones importantes")

        if max_duration > 300:
            validation["warnings"].append("Videos largos requieren más tiempo de procesamiento")

        # Recomendaciones
        if confidence > 0.6:
            validation["recommendations"].append("Para videos, considere usar confidence_threshold entre 0.3-0.5")

        if frame_skip == 1:
            validation["recommendations"].append("frame_skip=1 es lento, considere usar 3-5 para mejor rendimiento")

        validation["is_valid"] = len(validation["errors"]) == 0

    except Exception as e:
        validation["is_valid"] = False
        validation["errors"].append(f"Error validando request: {str(e)}")

    return validation


# Endpoint para testing de videos
@video_router.post("/test-pipeline",
                   summary="Test Pipeline de Video",
                   description="Endpoint de testing para verificar el pipeline completo de video")
async def test_video_pipeline(
        file: UploadFile = File(...),
        debug: Optional[bool] = Form(False),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """Endpoint de testing con información detallada para videos"""

    try:
        # Parámetros de testing para videos
        request_params = {
            "confidence_threshold": 0.3,  # Bajo para capturar más detecciones
            "iou_threshold": 0.4,
            "frame_skip": 2,  # Procesar más frames para testing
            "max_duration": 120,  # Máximo 2 minutos para testing
            "save_results": True,
            "save_best_frames": True,
            "create_annotated_video": False,
            "min_detection_frames": 1  # Permisivo para testing
        }

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
            "urls": result["result_urls"]
        }

        if debug:
            # Información adicional para debug
            test_response["debug_info"] = {
                "frame_skip_used": request_params["frame_skip"],
                "confidence_threshold": request_params["confidence_threshold"],
                "model_status": models.get_model_info(),
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