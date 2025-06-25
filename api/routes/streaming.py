"""
Rutas completas para streaming de video en tiempo real
Sistema integrado de WebSocket + procesamiento + monitoreo
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi import Depends, HTTPException, status, Query
from typing import Optional, Dict, Any
import asyncio
import json
import time
import uuid
from loguru import logger

from api.websocket_manager import connection_manager, StreamingStatus
from models import model_manager
from services.streaming_service import streaming_service
from services.file_service import file_service
from api.dependencies import get_model_manager, log_request_info
from models.model_manager import ModelManager
from config.settings import settings

# Router principal para streaming
streaming_router = APIRouter(prefix="/api/v1/streaming", tags=["🎬 Real-time Video Streaming"])


@streaming_router.websocket("/ws/{session_id}")
async def websocket_streaming_endpoint(websocket: WebSocket, session_id: str, request: Request):
    """
    🔌 WebSocket principal para streaming en tiempo real

    **Funcionalidades:**
    - Conexión y autenticación
    - Mensajes de control (pause/resume/stop)
    - Recepción de updates de procesamiento
    - Manejo de errores y desconexiones

    **Mensajes soportados:**
    - `ping` - Verificación de conectividad
    - `pause_processing` - Pausar procesamiento
    - `resume_processing` - Reanudar procesamiento
    - `stop_processing` - Detener procesamiento
    - `get_status` - Solicitar estado actual
    - `adjust_quality` - Ajustar calidad de streaming
    """
    client_ip = request.client.host if request.client else "unknown"

    try:
        # Intentar conectar cliente
        connected = await connection_manager.connect(websocket, session_id, client_ip)
        if not connected:
            logger.warning(f"🚫 Conexión rechazada: {session_id} desde {client_ip}")
            return

        logger.info(f"🔌 WebSocket streaming conectado: {session_id} desde {client_ip}")

        # Enviar configuración inicial
        await connection_manager.send_message(session_id, {
            "type": "streaming_config",
            "data": settings.get_streaming_config(),
            "session_info": {
                "session_id": session_id,
                "client_ip": client_ip,
                "server_time": time.time()
            }
        })

        # Loop principal de manejo de mensajes
        while True:
            try:
                # Recibir mensaje del cliente con timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)

                # Procesar comando del cliente
                await _handle_websocket_message(session_id, message)

            except asyncio.TimeoutError:
                # Enviar ping para mantener conexión viva
                await connection_manager.send_message(session_id, {
                    "type": "ping",
                    "timestamp": time.time()
                })
                continue

            except WebSocketDisconnect:
                logger.info(f"🔌 Cliente {session_id} desconectado normalmente")
                break

            except json.JSONDecodeError:
                logger.warning(f"⚠️ Mensaje JSON inválido de {session_id}")
                await connection_manager.send_message(session_id, {
                    "type": "error",
                    "error": {
                        "code": "INVALID_JSON",
                        "message": "Formato de mensaje inválido"
                    }
                })

            except Exception as e:
                logger.error(f"❌ Error en WebSocket {session_id}: {str(e)}")
                await connection_manager.send_message(session_id, {
                    "type": "error",
                    "error": {
                        "code": "WEBSOCKET_ERROR",
                        "message": f"Error interno: {str(e)}"
                    }
                })
                break

    except Exception as e:
        logger.error(f"❌ Error estableciendo conexión WebSocket {session_id}: {str(e)}")
    finally:
        await connection_manager.disconnect(session_id)
        logger.info(f"🔌 WebSocket {session_id} limpiado")


async def _handle_websocket_message(session_id: str, message: Dict[str, Any]):
    """
    Maneja mensajes recibidos del cliente WebSocket
    """
    message_type = message.get("type", "")
    message_data = message.get("data", {})

    try:
        if message_type == "ping":
            await connection_manager.send_message(session_id, {
                "type": "pong",
                "timestamp": time.time(),
                "session_id": session_id
            })

        elif message_type == "pause_processing":
            success = await streaming_service.pause_streaming(session_id)
            await connection_manager.send_message(session_id, {
                "type": "pause_response",
                "success": success,
                "timestamp": time.time()
            })

        elif message_type == "resume_processing":
            success = await streaming_service.resume_streaming(session_id)
            await connection_manager.send_message(session_id, {
                "type": "resume_response",
                "success": success,
                "timestamp": time.time()
            })

        elif message_type == "stop_processing":
            success = await streaming_service.stop_streaming(session_id)
            await connection_manager.send_message(session_id, {
                "type": "stop_response",
                "success": success,
                "timestamp": time.time()
            })

        elif message_type == "get_status":
            session = connection_manager.get_session(session_id)
            if session:
                status_data = {
                    "session_id": session_id,
                    "status": session.status.value,
                    "processed_frames": session.processed_frames,
                    "total_frames": session.total_frames,
                    "progress_percent": (session.processed_frames / max(session.total_frames, 1)) * 100,
                    "detections_count": len(session.detections),
                    "unique_plates_count": len(session.unique_plates),
                    "processing_speed": session.processing_speed,
                    "is_paused": session.is_paused,
                    "start_time": session.start_time,
                    "last_activity": session.last_activity
                }

                await connection_manager.send_message(session_id, {
                    "type": "status_update",
                    "data": status_data
                })
            else:
                await connection_manager.send_message(session_id, {
                    "type": "error",
                    "error": {
                        "code": "SESSION_NOT_FOUND",
                        "message": "Sesión no encontrada"
                    }
                })

        elif message_type == "adjust_quality":
            # Manejar ajuste de calidad manual
            quality = message_data.get("quality", 75)
            if 10 <= quality <= 100:
                # Ajustar calidad en el quality manager si existe
                if session_id in streaming_service.quality_managers:
                    streaming_service.quality_managers[session_id].current_quality = quality

                await connection_manager.send_message(session_id, {
                    "type": "quality_adjusted",
                    "data": {"new_quality": quality},
                    "timestamp": time.time()
                })
            else:
                await connection_manager.send_message(session_id, {
                    "type": "error",
                    "error": {
                        "code": "INVALID_QUALITY",
                        "message": "Calidad debe estar entre 10 y 100"
                    }
                })

        elif message_type == "get_detection_timeline":
            # Enviar timeline de detecciones
            if session_id in streaming_service.detection_trackers:
                tracker = streaming_service.detection_trackers[session_id]
                timeline = tracker.detection_timeline[-50:]  # Últimas 50 detecciones

                await connection_manager.send_message(session_id, {
                    "type": "detection_timeline",
                    "data": timeline,
                    "count": len(timeline)
                })

        elif message_type == "seek_to_frame":
            # Solicitar saltar a un frame específico (para implementación futura)
            frame_num = message_data.get("frame_number", 0)
            await connection_manager.send_message(session_id, {
                "type": "seek_response",
                "data": {
                    "requested_frame": frame_num,
                    "status": "not_implemented"
                }
            })

        elif message_type == "get_metrics":
            # Enviar métricas del sistema
            metrics = {
                "connection_metrics": connection_manager.metrics.get_stats(),
                "streaming_metrics": streaming_service.get_streaming_stats(),
                "system_info": {
                    "active_sessions": len(connection_manager.sessions),
                    "max_sessions": settings.max_websocket_connections,
                    "streaming_enabled": settings.streaming_enabled
                }
            }

            await connection_manager.send_message(session_id, {
                "type": "metrics_update",
                "data": metrics
            })

        else:
            logger.warning(f"⚠️ Tipo de mensaje desconocido de {session_id}: {message_type}")
            await connection_manager.send_message(session_id, {
                "type": "error",
                "error": {
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "message": f"Tipo de mensaje no soportado: {message_type}",
                    "supported_types": [
                        "ping", "pause_processing", "resume_processing",
                        "stop_processing", "get_status", "adjust_quality",
                        "get_detection_timeline", "seek_to_frame", "get_metrics"
                    ]
                }
            })

    except Exception as e:
        logger.error(f"❌ Error manejando mensaje {message_type} de {session_id}: {str(e)}")
        await connection_manager.send_message(session_id, {
            "type": "error",
            "error": {
                "code": "MESSAGE_PROCESSING_ERROR",
                "message": f"Error procesando mensaje: {str(e)}"
            }
        })


@streaming_router.post("/start-session",
                       summary="🚀 Iniciar Sesión de Streaming",
                       description="""
                       Inicia una nueva sesión de streaming en tiempo real.

                       **Proceso:**
                       1. Valida archivo de video
                       2. Crea sesión WebSocket
                       3. Inicia procesamiento en background
                       4. Envía updates vía WebSocket

                       **Requisitos:**
                       - WebSocket debe estar conectado primero
                       - Video en formato soportado (MP4, AVI, MOV, MKV, WebM)
                       - Tamaño máximo: 150MB
                       """)
async def start_streaming_session(
        session_id: str = Form(..., description="ID de sesión WebSocket (debe estar conectado)"),
        file: UploadFile = File(..., description="Video a procesar"),
        confidence_threshold: Optional[float] = Form(0.3, ge=0.1, le=1.0, description="Umbral de confianza"),
        iou_threshold: Optional[float] = Form(0.4, ge=0.1, le=1.0, description="Umbral IoU"),
        frame_skip: Optional[int] = Form(2, ge=1, le=10, description="Procesar cada N frames"),
        max_duration: Optional[int] = Form(600, ge=10, le=1800, description="Duración máxima en segundos"),
        send_all_frames: Optional[bool] = Form(False, description="Enviar todos los frames o solo con detecciones"),
        adaptive_quality: Optional[bool] = Form(True, description="Activar calidad adaptativa"),
        enable_thumbnails: Optional[bool] = Form(True, description="Generar thumbnails pequeños"),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """Inicia una sesión de streaming de video en tiempo real"""

    try:
        logger.info(f"🚀 Iniciando sesión de streaming: {session_id}")

        # Validar que la sesión WebSocket existe
        session = connection_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "WEBSOCKET_SESSION_NOT_FOUND",
                    "message": f"Sesión WebSocket {session_id} no encontrada",
                    "hint": "Conecta primero vía WebSocket a /api/v1/streaming/ws/{session_id}"
                }
            )

        if not session.websocket:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "WEBSOCKET_NOT_ACTIVE",
                    "message": "Sesión WebSocket no está activa"
                }
            )

        # Validar que los modelos estén cargados
        if not models.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "MODELS_NOT_LOADED",
                    "message": "Los modelos ALPR no están cargados"
                }
            )

        # Validar archivo de video
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "NO_FILENAME",
                    "message": "Nombre de archivo requerido"
                }
            )

        # Verificar extensión
        video_extensions = settings.video_extensions_list
        file_extension = file.filename.split('.')[-1].lower()

        if file_extension not in video_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "UNSUPPORTED_FORMAT",
                    "message": f"Formato no soportado para streaming",
                    "supported_formats": video_extensions
                }
            )

        # Guardar archivo
        file_path, file_info = await file_service.save_upload_file(file, "streaming_")

        # Configurar parámetros de procesamiento
        processing_params = {
            "session_id": session_id,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "frame_skip": frame_skip,
            "max_duration": max_duration,
            "send_all_frames": send_all_frames,
            "adaptive_quality": adaptive_quality,
            "enable_thumbnails": enable_thumbnails,
            "streaming_mode": True
        }

        # Actualizar estado de sesión
        connection_manager.update_session_status(session_id, StreamingStatus.INITIALIZING)

        # Enviar confirmación inmediata
        await connection_manager.send_message(session_id, {
            "type": "session_accepted",
            "data": {
                "session_id": session_id,
                "file_info": file_info,
                "processing_params": processing_params,
                "streaming_config": settings.get_streaming_config()
            }
        })

        # Iniciar streaming en background
        success = await streaming_service.start_video_streaming(
            session_id, file_path, file_info, processing_params
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "STREAMING_START_FAILED",
                    "message": "No se pudo iniciar el streaming"
                }
            )

        logger.success(f"✅ Streaming iniciado: {session_id} - {file_info['filename']}")

        return {
            "success": True,
            "message": "Sesión de streaming iniciada exitosamente",
            "data": {
                "session_id": session_id,
                "file_info": {
                    "filename": file_info["filename"],
                    "size_mb": file_info["size_mb"],
                    "file_type": file_info.get("file_type", "video")
                },
                "processing_params": processing_params,
                "websocket_status": "connected",
                "streaming_status": "initializing"
            },
            "timestamp": time.time()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error iniciando streaming {session_id}: {str(e)}")

        # Notificar error vía WebSocket si es posible
        try:
            await connection_manager.send_message(session_id, {
                "type": "session_error",
                "error": {
                    "code": "SESSION_START_ERROR",
                    "message": str(e)
                }
            })
        except:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Error interno iniciando streaming: {str(e)}"
            }
        )


@streaming_router.get("/sessions",
                      summary="📋 Listar Sesiones Activas",
                      description="Obtiene información de todas las sesiones de streaming activas")
async def get_streaming_sessions(
        include_details: Optional[bool] = Query(False, description="Incluir detalles completos"),
        status_filter: Optional[str] = Query(None, description="Filtrar por estado"),
        request_id: str = Depends(log_request_info)
):
    """Lista todas las sesiones de streaming activas"""

    try:
        sessions_info = connection_manager.get_active_sessions_info()

        # Aplicar filtro de estado si se especifica
        if status_filter:
            filtered_sessions = {}
            for session_id, session_data in sessions_info["sessions"].items():
                if session_data.get("status") == status_filter:
                    filtered_sessions[session_id] = session_data
            sessions_info["sessions"] = filtered_sessions

        # Reducir información si no se solicitan detalles
        if not include_details:
            sessions_info.pop("system_metrics", None)
            for session_data in sessions_info["sessions"].values():
                # Mantener solo info esencial
                essential_keys = [
                    "session_id", "status", "progress_percent",
                    "unique_plates_count", "processing_speed"
                ]
                session_data = {k: v for k, v in session_data.items() if k in essential_keys}

        # Agregar estadísticas de streaming
        streaming_stats = streaming_service.get_streaming_stats()

        return {
            "success": True,
            "data": {
                **sessions_info,
                "streaming_stats": streaming_stats
            },
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo sesiones: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo información de sesiones: {str(e)}"
        )


@streaming_router.get("/sessions/{session_id}",
                      summary="📄 Información de Sesión Específica",
                      description="Obtiene información detallada de una sesión de streaming")
async def get_session_details(
        session_id: str,
        include_timeline: Optional[bool] = Query(False, description="Incluir timeline de detecciones"),
        include_frames: Optional[bool] = Query(False, description="Incluir información de frames"),
        request_id: str = Depends(log_request_info)
):
    """Obtiene información detallada de una sesión específica"""

    try:
        session = connection_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sesión {session_id} no encontrada"
            )

        current_time = time.time()

        # Información básica de la sesión
        session_details = {
            "session_id": session_id,
            "status": session.status.value,
            "connected": session.websocket is not None,

            # Información del archivo
            "file_info": session.file_info,
            "video_info": session.video_info,

            # Progreso
            "progress": {
                "total_frames": session.total_frames,
                "processed_frames": session.processed_frames,
                "current_frame": session.current_frame,
                "progress_percent": (
                    (session.processed_frames / max(session.total_frames, 1)) * 100
                    if session.total_frames > 0 else 0
                )
            },

            # Detecciones
            "detection_summary": {
                "total_detections": len(session.detections),
                "unique_plates": len(session.unique_plates),
                "frames_with_detections": session.frames_with_detections,
                "best_detection": session.best_detection
            },

            # Timing
            "timing": {
                "start_time": session.start_time,
                "uptime": current_time - session.start_time if session.start_time else 0,
                "last_activity": session.last_activity,
                "processing_speed": session.processing_speed
            },

            # Control
            "control": {
                "is_paused": session.is_paused,
                "should_stop": session.should_stop
            },

            # Parámetros
            "processing_params": session.processing_params
        }

        # Incluir timeline si se solicita
        if include_timeline and session_id in streaming_service.detection_trackers:
            tracker = streaming_service.detection_trackers[session_id]
            session_details["detection_timeline"] = tracker.detection_timeline[-100:]

        # Incluir información de frames si se solicita
        if include_frames and session_id in streaming_service.detection_trackers:
            tracker = streaming_service.detection_trackers[session_id]
            session_details["frame_detections"] = dict(list(tracker.frame_detections.items())[-20:])

        # Información de calidad si existe
        if session_id in streaming_service.quality_managers:
            quality_manager = streaming_service.quality_managers[session_id]
            session_details["quality_info"] = {
                "current_quality": quality_manager.current_quality,
                "quality_history": list(quality_manager.quality_history),
                "adaptive_enabled": settings.streaming_adaptive_quality
            }

        return {
            "success": True,
            "data": session_details,
            "timestamp": current_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error obteniendo detalles de sesión {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo información: {str(e)}"
        )


@streaming_router.post("/sessions/{session_id}/control",
                       summary="🎮 Control de Sesión",
                       description="Controla una sesión de streaming (pause/resume/stop)")
async def control_streaming_session(
        session_id: str,
        action: str = Form(..., description="Acción: pause, resume, stop"),
        force: Optional[bool] = Form(False, description="Forzar acción"),
        request_id: str = Depends(log_request_info)
):
    """Controla una sesión de streaming"""

    try:
        session = connection_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sesión {session_id} no encontrada"
            )

        valid_actions = ["pause", "resume", "stop"]
        if action not in valid_actions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Acción inválida. Acciones válidas: {', '.join(valid_actions)}"
            )

        success = False
        message = ""

        if action == "pause":
            if session.status != StreamingStatus.PROCESSING and not force:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="La sesión no está en procesamiento. Usa force=true para forzar."
                )
            success = await streaming_service.pause_streaming(session_id)
            message = "Sesión pausada" if success else "No se pudo pausar la sesión"

        elif action == "resume":
            if session.status != StreamingStatus.PAUSED and not force:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="La sesión no está pausada. Usa force=true para forzar."
                )
            success = await streaming_service.resume_streaming(session_id)
            message = "Sesión reanudada" if success else "No se pudo reanudar la sesión"

        elif action == "stop":
            success = await streaming_service.stop_streaming(session_id)
            message = "Sesión detenida" if success else "No se pudo detener la sesión"

        if not success and not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )

        return {
            "success": True,
            "message": message,
            "data": {
                "session_id": session_id,
                "action": action,
                "previous_status": session.status.value,
                "new_status": session.status.value,
                "forced": force
            },
            "timestamp": time.time()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error controlando sesión {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error controlando sesión: {str(e)}"
        )


@streaming_router.delete("/sessions/{session_id}",
                         summary="🗑️ Eliminar Sesión",
                         description="Elimina una sesión de streaming y limpia recursos")
async def delete_streaming_session(
        session_id: str,
        force: Optional[bool] = Query(False, description="Forzar eliminación"),
        cleanup_files: Optional[bool] = Query(True, description="Limpiar archivos temporales"),
        request_id: str = Depends(log_request_info)
):
    """Elimina una sesión de streaming"""

    try:
        session = connection_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sesión {session_id} no encontrada"
            )

        # Verificar si se puede eliminar
        if session.status == StreamingStatus.PROCESSING and not force:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Sesión en procesamiento. Usa force=true para eliminar forzosamente"
            )

        # Detener procesamiento si está activo
        if session.status in [StreamingStatus.PROCESSING, StreamingStatus.PAUSED]:
            await streaming_service.stop_streaming(session_id)

        # Limpiar archivos si se solicita
        files_cleaned = []
        if cleanup_files and session.video_path:
            try:
                file_service.cleanup_temp_file(session.video_path)
                files_cleaned.append(session.video_path)
            except Exception as e:
                logger.warning(f"⚠️ Error limpiando archivo {session.video_path}: {str(e)}")

        # Desconectar sesión
        await connection_manager.disconnect(session_id)

        logger.info(f"🗑️ Sesión eliminada: {session_id}")

        return {
            "success": True,
            "message": f"Sesión {session_id} eliminada exitosamente",
            "data": {
                "session_id": session_id,
                "files_cleaned": files_cleaned,
                "forced": force
            },
            "timestamp": time.time()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error eliminando sesión {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error eliminando sesión: {str(e)}"
        )


@streaming_router.get("/health",
                      summary="🏥 Estado del Sistema de Streaming",
                      description="Verifica el estado completo del sistema de streaming")
async def streaming_health_check():
    """Health check completo del sistema de streaming"""

    try:
        # Métricas del sistema
        system_metrics = connection_manager.metrics.get_stats()
        sessions_info = connection_manager.get_active_sessions_info()
        streaming_stats = streaming_service.get_streaming_stats()

        # Estado de modelos
        models_status = model_manager.get_model_info()

        # Validar configuración de streaming
        streaming_config_valid = settings.validate_streaming_settings()

        # Calcular estado general
        total_sessions = sessions_info["total_sessions"]
        active_connections = sessions_info["active_connections"]
        max_sessions = settings.max_websocket_connections

        health_status = "healthy"
        issues = []

        if total_sessions > max_sessions * 0.8:  # Más del 80% de capacidad
            health_status = "warning"
            issues.append("Acercándose al límite de sesiones")

        if system_metrics["websocket_errors"] > 20:
            health_status = "degraded"
            issues.append("Muchos errores de WebSocket")

        if not models_status["models_loaded"]:
            health_status = "error"
            issues.append("Modelos ALPR no cargados")

        if not streaming_config_valid["all_valid"]:
            health_status = "warning"
            issues.append("Configuración de streaming subóptima")

        return {
            "status": health_status,
            "timestamp": time.time(),
            "service": "CARID Streaming Service",
            "version": settings.app_version,
            "issues": issues,

            "connections": {
                "total_sessions": total_sessions,
                "active_connections": active_connections,
                "max_sessions": max_sessions,
                "capacity_usage_percent": (total_sessions / max_sessions) * 100
            },

            "models": {
                "loaded": models_status["models_loaded"],
                "device": models_status["device"],
                "plate_detector": models_status["plate_detector_loaded"],
                "char_recognizer": models_status["char_recognizer_loaded"]
            },

            "system_metrics": system_metrics,
            "streaming_stats": streaming_stats,

            "capabilities": {
                "websocket_streaming": True,
                "real_time_processing": True,
                "adaptive_quality": settings.streaming_adaptive_quality,
                "multi_session": True,
                "pause_resume_control": True,
                "frame_visualization": True,
                "detection_timeline": True
            },

            "configuration": {
                "streaming_enabled": settings.streaming_enabled,
                "max_file_size_mb": settings.max_file_size,
                "max_video_duration": settings.max_video_duration,
                "supported_formats": settings.video_extensions_list,
                "frame_quality": settings.streaming_frame_quality,
                "adaptive_quality": settings.streaming_adaptive_quality
            }
        }

    except Exception as e:
        logger.error(f"❌ Error en health check de streaming: {str(e)}")
        return {
            "status": "error",
            "timestamp": time.time(),
            "service": "CARID Streaming Service",
            "error": str(e)
        }


@streaming_router.get("/config",
                      summary="⚙️ Configuración del Sistema",
                      description="Obtiene configuración completa del sistema de streaming")
async def get_streaming_configuration():
    """Obtiene la configuración completa del sistema de streaming"""

    try:
        streaming_config = settings.get_streaming_config()
        video_config = settings.get_video_processing_config()

        return {
            "success": True,
            "data": {
                "streaming": streaming_config,
                "video_processing": video_config,
                "file_limits": {
                    "max_file_size_mb": settings.max_file_size,
                    "max_video_duration": settings.max_video_duration,
                    "supported_formats": settings.video_extensions_list
                },
                "performance": {
                    "device": settings.device,
                    "cuda_available": settings.is_cuda_available,
                    "max_concurrent_sessions": settings.max_websocket_connections
                },
                "features": {
                    "adaptive_quality": settings.streaming_adaptive_quality,
                    "compression": settings.streaming_compression_enabled,
                    "throttling": settings.streaming_throttle_enabled,
                    "thumbnails": True,
                    "timeline_tracking": True
                }
            },
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo configuración: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo configuración: {str(e)}"
        )


@streaming_router.post("/test-connection",
                       summary="🧪 Probar Conexión",
                       description="Prueba la conectividad y capacidades del sistema")
async def test_streaming_connection():
    """Prueba la conectividad y funcionalidad del sistema de streaming"""

    try:
        test_results = {
            "timestamp": time.time(),
            "test_id": str(uuid.uuid4())[:8],
            "tests": {}
        }

        # Test 1: Estado de modelos
        models_status = model_manager.get_model_info()
        test_results["tests"]["models"] = {
            "status": "pass" if models_status["models_loaded"] else "fail",
            "details": models_status
        }

        # Test 2: Configuración de streaming
        streaming_config = settings.validate_streaming_settings()
        test_results["tests"]["streaming_config"] = {
            "status": "pass" if streaming_config["all_valid"] else "warn",
            "details": streaming_config
        }

        # Test 3: Capacidad del sistema
        sessions_info = connection_manager.get_active_sessions_info()
        capacity_test = sessions_info["total_sessions"] < settings.max_websocket_connections
        test_results["tests"]["capacity"] = {
            "status": "pass" if capacity_test else "warn",
            "details": {
                "current_sessions": sessions_info["total_sessions"],
                "max_sessions": settings.max_websocket_connections,
                "available_slots": settings.max_websocket_connections - sessions_info["total_sessions"]
            }
        }

        # Test 4: WebSocket manager
        test_results["tests"]["websocket_manager"] = {
            "status": "pass",
            "details": {
                "active_connections": sessions_info["active_connections"],
                "security_enabled": True,
                "metrics_enabled": True
            }
        }

        # Calcular resultado general
        passed_tests = sum(1 for test in test_results["tests"].values() if test["status"] == "pass")
        total_tests = len(test_results["tests"])

        test_results["summary"] = {
            "overall_status": "pass" if passed_tests == total_tests else "warn",
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": (passed_tests / total_tests) * 100
        }

        return {
            "success": True,
            "message": "Pruebas de conectividad completadas",
            "data": test_results
        }

    except Exception as e:
        logger.error(f"❌ Error en test de conexión: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en pruebas: {str(e)}"
        )


# WebSocket de prueba simple
@streaming_router.websocket("/test-ws/{test_session_id}")
async def test_websocket_connection(websocket: WebSocket, test_session_id: str):
    """WebSocket de prueba para verificar conectividad básica"""

    try:
        await websocket.accept()

        await websocket.send_text(json.dumps({
            "type": "test_connection_established",
            "message": "Conexión WebSocket de prueba establecida exitosamente",
            "session_id": test_session_id,
            "timestamp": time.time(),
            "server_info": {
                "service": "CARID Streaming Service",
                "version": settings.app_version,
                "streaming_enabled": settings.streaming_enabled
            }
        }))

        # Loop de prueba
        ping_counter = 0
        while True:
            try:
                # Enviar ping cada 10 segundos
                await asyncio.sleep(10)
                ping_counter += 1

                await websocket.send_text(json.dumps({
                    "type": "test_ping",
                    "counter": ping_counter,
                    "message": f"Test ping #{ping_counter}",
                    "timestamp": time.time()
                }))

                # Intentar recibir respuesta
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
                    message = json.loads(data)

                    await websocket.send_text(json.dumps({
                        "type": "test_echo",
                        "received_message": message,
                        "echo_timestamp": time.time()
                    }))

                except asyncio.TimeoutError:
                    # No hay respuesta del cliente, continuar
                    pass

            except WebSocketDisconnect:
                logger.info(f"🔌 Test WebSocket {test_session_id} desconectado")
                break

    except Exception as e:
        logger.error(f"❌ Error en test WebSocket {test_session_id}: {str(e)}")
    finally:
        logger.info(f"🧪 Test WebSocket {test_session_id} finalizado")


@streaming_router.get("/stats",
                      summary="📊 Estadísticas Completas",
                      description="Estadísticas detalladas del sistema de streaming")
async def get_comprehensive_stats(
        include_history: Optional[bool] = Query(False, description="Incluir historial de métricas"),
        session_filter: Optional[str] = Query(None, description="Filtrar por sesión específica")
):
    """Obtiene estadísticas comprensivas del sistema de streaming"""

    try:
        # Estadísticas básicas
        sessions_info = connection_manager.get_active_sessions_info()
        system_metrics = connection_manager.metrics.get_stats()
        streaming_stats = streaming_service.get_streaming_stats()

        stats = {
            "timestamp": time.time(),
            "overview": {
                "total_sessions": sessions_info["total_sessions"],
                "active_connections": sessions_info["active_connections"],
                "system_uptime": system_metrics["total_time"],
                "service_version": settings.app_version
            },

            "performance": {
                "messages_per_second": system_metrics["messages_per_second"],
                "bandwidth_mbps": system_metrics["bandwidth_mbps"],
                "websocket_errors": system_metrics["websocket_errors"],
                "avg_processing_time": system_metrics["avg_processing_time"]
            },

            "streaming": {
                "frames_sent": system_metrics["frames_sent"],
                "active_quality_managers": streaming_stats["quality_managers"],
                "active_trackers": streaming_stats["detection_trackers"],
                "streaming_config": streaming_stats["streaming_config"]
            },

            "capacity": {
                "max_sessions": settings.max_websocket_connections,
                "available_slots": settings.max_websocket_connections - sessions_info["total_sessions"],
                "capacity_usage_percent": (sessions_info["total_sessions"] / settings.max_websocket_connections) * 100
            }
        }

        # Filtrar por sesión específica si se solicita
        if session_filter:
            if session_filter in sessions_info["sessions"]:
                stats["filtered_session"] = sessions_info["sessions"][session_filter]
            else:
                stats["filtered_session"] = None
                stats["filter_error"] = f"Sesión {session_filter} no encontrada"

        # Incluir detalles de sesiones
        stats["sessions"] = sessions_info["sessions"]

        # Incluir historial si se solicita
        if include_history:
            # TODO: Implementar sistema de historial de métricas
            stats["history"] = {
                "note": "Sistema de historial en desarrollo",
                "available": False
            }

        return {
            "success": True,
            "data": stats,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas comprensivas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo estadísticas: {str(e)}"
        )


@streaming_router.post("/sessions/cleanup",
                       summary="🧹 Limpiar Sesiones",
                       description="Limpia sesiones inactivas y libera recursos")
async def cleanup_streaming_sessions(
        max_age_minutes: Optional[int] = Form(30, description="Edad máxima en minutos"),
        force: Optional[bool] = Form(False, description="Forzar limpieza"),
        cleanup_files: Optional[bool] = Form(True, description="Limpiar archivos temporales"),
        request_id: str = Depends(log_request_info)
):
    """Limpia sesiones inactivas manualmente"""

    try:
        current_time = time.time()
        max_age_seconds = max_age_minutes * 60

        sessions_to_cleanup = []
        cleanup_reasons = {}

        for session_id, session in connection_manager.sessions.items():
            reason = None

            if force:
                reason = "forced_cleanup"
            elif session.last_activity:
                age = current_time - session.last_activity
                if age > max_age_seconds:
                    reason = f"inactive_for_{age / 60:.1f}_minutes"
            elif session.status in [StreamingStatus.ERROR, StreamingStatus.COMPLETED]:
                if session.start_time:
                    age = current_time - session.start_time
                    if age > max_age_seconds:
                        reason = f"completed_or_error_for_{age / 60:.1f}_minutes"

            if reason:
                sessions_to_cleanup.append(session_id)
                cleanup_reasons[session_id] = reason

        # Limpiar sesiones identificadas
        cleaned_sessions = []
        files_cleaned = []
        errors = []

        for session_id in sessions_to_cleanup:
            try:
                session = connection_manager.get_session(session_id)

                # Limpiar archivos si se solicita
                if cleanup_files and session and session.video_path:
                    try:
                        file_service.cleanup_temp_file(session.video_path)
                        files_cleaned.append(session.video_path)
                    except Exception as e:
                        errors.append(f"Error limpiando archivo {session.video_path}: {str(e)}")

                # Desconectar sesión
                await connection_manager.disconnect(session_id)

                cleaned_sessions.append({
                    "session_id": session_id,
                    "reason": cleanup_reasons[session_id]
                })

            except Exception as e:
                errors.append(f"Error limpiando sesión {session_id}: {str(e)}")

        return {
            "success": True,
            "message": f"Limpieza completada: {len(cleaned_sessions)} sesiones eliminadas",
            "data": {
                "sessions_cleaned": cleaned_sessions,
                "files_cleaned": files_cleaned,
                "sessions_checked": len(connection_manager.sessions) + len(cleaned_sessions),
                "errors": errors,
                "criteria": {
                    "max_age_minutes": max_age_minutes,
                    "force": force,
                    "cleanup_files": cleanup_files
                }
            },
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"❌ Error en limpieza de sesiones: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en limpieza: {str(e)}"
        )


@streaming_router.get("/sessions/{session_id}/download",
                      summary="💾 Descargar Resultados",
                      description="Descarga resultados de una sesión en formato JSON o CSV")
async def download_session_results(
        session_id: str,
        format: Optional[str] = Query("json", description="Formato: json, csv"),
        include_timeline: Optional[bool] = Query(True, description="Incluir timeline de detecciones"),
        include_frames: Optional[bool] = Query(False, description="Incluir información de frames"),
        compression: Optional[bool] = Query(False, description="Comprimir archivo")
):
    """Descarga los resultados de una sesión"""

    try:
        session = connection_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sesión {session_id} no encontrada"
            )

        # Preparar datos base
        export_data = {
            "session_info": {
                "session_id": session_id,
                "status": session.status.value,
                "start_time": session.start_time,
                "processing_params": session.processing_params,
                "file_info": session.file_info,
                "video_info": session.video_info
            },
            "processing_summary": {
                "total_frames": session.total_frames,
                "processed_frames": session.processed_frames,
                "frames_with_detections": session.frames_with_detections,
                "total_detections": len(session.detections),
                "unique_plates_count": len(session.unique_plates),
                "processing_speed": session.processing_speed
            },
            "detections": session.detections,
            "unique_plates": session.unique_plates,
            "best_detection": session.best_detection
        }

        # Incluir timeline si se solicita
        if include_timeline and session_id in streaming_service.detection_trackers:
            tracker = streaming_service.detection_trackers[session_id]
            export_data["detection_timeline"] = tracker.detection_timeline

        # Incluir frames si se solicita
        if include_frames and session_id in streaming_service.detection_trackers:
            tracker = streaming_service.detection_trackers[session_id]
            export_data["frame_detections"] = tracker.frame_detections

        # Generar estadísticas
        if session.detections:
            confidences = [d["overall_confidence"] for d in session.detections]
            export_data["statistics"] = {
                "confidence_stats": {
                    "avg": sum(confidences) / len(confidences),
                    "max": max(confidences),
                    "min": min(confidences),
                    "count": len(confidences)
                },
                "valid_plates": len([d for d in session.detections if d.get("is_valid_plate", False)]),
                "export_timestamp": time.time()
            }

        # Exportar según formato
        if format.lower() == "json":
            from fastapi.responses import JSONResponse

            filename = f"carid_streaming_results_{session_id}_{int(time.time())}.json"

            return JSONResponse(
                content=export_data,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Content-Type": "application/json"
                }
            )

        elif format.lower() == "csv":
            if not session.detections:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No hay detecciones para exportar a CSV"
                )

            import pandas as pd
            from fastapi.responses import StreamingResponse
            import io

            # Convertir detecciones a DataFrame
            df = pd.DataFrame(session.detections)

            # Generar CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            filename = f"carid_detections_{session_id}_{int(time.time())}.csv"

            return StreamingResponse(
                io.BytesIO(csv_buffer.getvalue().encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Formato no soportado. Usar 'json' o 'csv'"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error descargando resultados {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generando descarga: {str(e)}"
        )


# Middleware para logging específico de streaming
@streaming_router.middleware("http")
async def log_streaming_requests(request: Request, call_next):
    """Middleware para logging específico de streaming"""

    start_time = time.time()

    # Log de request entrante
    if request.url.path.startswith("/api/v1/streaming"):
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"🌐 Streaming request: {request.method} {request.url.path} from {client_ip}")

    response = await call_next(request)

    # Log de response
    process_time = time.time() - start_time
    if request.url.path.startswith("/api/v1/streaming"):
        logger.info(f"✅ Streaming response: {response.status_code} in {process_time:.3f}s")

    return response