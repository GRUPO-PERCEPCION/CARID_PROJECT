"""
Rutas para streaming de video en tiempo real - Sistema completo
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi import Depends, HTTPException, status
from typing import Optional
import asyncio
import json
import time
from loguru import logger

from api.websocket_manager import connection_manager, StreamingStatus
from services.real_time_video_service import real_time_video_service
from api.dependencies import get_model_manager, log_request_info
from models.model_manager import ModelManager
from config.settings import settings

streaming_router = APIRouter(prefix="/api/v1/streaming", tags=["Real-time Streaming"])


@streaming_router.websocket("/video/{session_id}")
async def websocket_video_streaming(websocket: WebSocket, session_id: str, request: Request):
    """
    WebSocket endpoint para streaming de procesamiento de video en tiempo real

    Maneja:
    - Conexión y autenticación
    - Mensajes de control (pause/resume/stop)
    - Envío de datos de procesamiento
    - Manejo de errores y desconexiones
    """
    client_ip = request.client.host if request.client else "unknown"

    try:
        # Intentar conectar cliente
        connected = await connection_manager.connect(websocket, session_id, client_ip)
        if not connected:
            return

        logger.info(f"🔌 WebSocket conectado: {session_id} desde {client_ip}")

        # Loop principal de manejo de mensajes
        while True:
            try:
                # Recibir mensaje del cliente
                data = await websocket.receive_text()
                message = json.loads(data)

                # Procesar comando del cliente
                await handle_client_message(session_id, message)

            except WebSocketDisconnect:
                logger.info(f"🔌 Cliente {session_id} desconectado normalmente")
                break
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Mensaje JSON inválido de {session_id}")
                await connection_manager.send_message(session_id, {
                    "type": "error",
                    "error": "Formato de mensaje inválido"
                })
            except Exception as e:
                logger.error(f"❌ Error en WebSocket {session_id}: {str(e)}")
                await connection_manager.send_message(session_id, {
                    "type": "error",
                    "error": f"Error interno: {str(e)}"
                })
                break

    except Exception as e:
        logger.error(f"❌ Error estableciendo conexión WebSocket {session_id}: {str(e)}")
    finally:
        await connection_manager.disconnect(session_id)


async def handle_client_message(session_id: str, message: Dict[str, Any]):
    """
    Maneja mensajes recibidos del cliente WebSocket

    Tipos de mensaje soportados:
    - ping: Verificación de conectividad
    - pause_processing: Pausar procesamiento
    - resume_processing: Reanudar procesamiento
    - stop_processing: Detener procesamiento
    - get_status: Solicitar estado actual
    - get_metrics: Solicitar métricas de rendimiento
    """
    message_type = message.get("type", "")

    try:
        if message_type == "ping":
            await connection_manager.send_message(session_id, {
                "type": "pong",
                "timestamp": time.time(),
                "session_id": session_id
            })

        elif message_type == "pause_processing":
            success = await connection_manager.pause_session(session_id)
            if success:
                await real_time_video_service.pause_processing(session_id)

        elif message_type == "resume_processing":
            success = await connection_manager.resume_session(session_id)
            if success:
                await real_time_video_service.resume_processing(session_id)

        elif message_type == "stop_processing":
            success = await connection_manager.stop_session(session_id)
            if success:
                await real_time_video_service.stop_processing(session_id)

        elif message_type == "get_status":
            session = connection_manager.get_session(session_id)
            if session:
                await connection_manager.send_message(session_id, {
                    "type": "status_update",
                    "data": {
                        "status": session.status.value,
                        "processed_frames": session.processed_frames,
                        "total_frames": session.total_frames,
                        "detections_count": len(session.detections),
                        "unique_plates_count": len(session.unique_plates),
                        "processing_speed": session.processing_speed,
                        "is_paused": session.is_paused
                    }
                })

        elif message_type == "get_metrics":
            metrics = connection_manager.metrics.get_stats()
            await connection_manager.send_message(session_id, {
                "type": "metrics_update",
                "data": metrics
            })

        else:
            logger.warning(f"⚠️ Tipo de mensaje desconocido de {session_id}: {message_type}")
            await connection_manager.send_message(session_id, {
                "type": "error",
                "error": f"Tipo de mensaje no soportado: {message_type}"
            })

    except Exception as e:
        logger.error(f"❌ Error manejando mensaje {message_type} de {session_id}: {str(e)}")
        await connection_manager.send_message(session_id, {
            "type": "error",
            "error": f"Error procesando mensaje: {str(e)}"
        })


@streaming_router.post("/start-processing",
                       summary="Iniciar Procesamiento en Tiempo Real",
                       description="""
                      Inicia el procesamiento de video en tiempo real con streaming vía WebSocket.

                      **Requisitos:**
                      - Sesión WebSocket ya establecida
                      - Archivo de video válido (MP4, AVI, MOV, MKV, WebM)
                      - Tamaño máximo según configuración (150MB por defecto)

                      **Proceso:**
                      1. Valida la sesión WebSocket
                      2. Guarda el archivo temporalmente
                      3. Inicia procesamiento en background
                      4. Envía updates vía WebSocket
                      """)
async def start_video_processing(
        session_id: str = Form(..., description="ID de sesión WebSocket"),
        file: UploadFile = File(..., description="Archivo de video a procesar"),
        confidence_threshold: Optional[float] = Form(0.3, ge=0.1, le=1.0, description="Umbral de confianza"),
        iou_threshold: Optional[float] = Form(0.4, ge=0.1, le=1.0, description="Umbral IoU"),
        frame_skip: Optional[int] = Form(2, ge=1, le=10, description="Procesar cada N frames"),
        max_duration: Optional[int] = Form(600, ge=10, le=1800, description="Duración máxima en segundos"),
        enhance_processing: Optional[bool] = Form(False, description="Aplicar mejoras de imagen"),
        save_best_frames: Optional[bool] = Form(True, description="Guardar frames con mejores detecciones"),
        request_id: str = Depends(log_request_info),
        models: ModelManager = Depends(get_model_manager)
):
    """Inicia el procesamiento de video en tiempo real"""

    try:
        logger.info(f"🎬 Iniciando procesamiento para sesión: {session_id}")

        # Validar que la sesión WebSocket existe y está conectada
        session = connection_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sesión {session_id} no encontrada"
            )

        current_time = time.time()

        session_info = {
            "session_id": session_id,
            "status": session.status.value,
            "connected": session.websocket is not None,
            "file_info": session.file_info,
            "video_info": session.video_info,
            "progress": {
                "total_frames": session.total_frames,
                "processed_frames": session.processed_frames,
                "current_frame": session.current_frame,
                "progress_percent": (
                    (session.processed_frames / max(session.total_frames, 1)) * 100
                    if session.total_frames > 0 else 0
                )
            },
            "detection_summary": {
                "total_detections": len(session.detections),
                "unique_plates": len(session.unique_plates),
                "frames_with_detections": session.frames_with_detections,
                "best_detection": session.best_detection
            },
            "timing": {
                "start_time": session.start_time,
                "uptime": current_time - session.start_time if session.start_time else 0,
                "last_activity": session.last_activity,
                "processing_speed": session.processing_speed
            },
            "control": {
                "is_paused": session.is_paused,
                "should_stop": session.should_stop
            },
            "processing_params": session.processing_params
        }

        # Incluir detecciones si se solicita
        if include_detections:
            session_info["detections"] = session.detections
            session_info["unique_plates_detailed"] = session.unique_plates

        return {
            "success": True,
            "data": session_info,
            "timestamp": current_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error obteniendo info de sesión {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo información: {str(e)}"
        )


@streaming_router.post("/sessions/{session_id}/control",
                       summary="Controlar Sesión de Streaming")
async def control_session(
        session_id: str,
        action: str = Form(..., description="Acción: pause, resume, stop"),
        request_id: str = Depends(log_request_info)
):
    """Controla una sesión de streaming (pausar, reanudar, detener)"""

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
            success = await connection_manager.pause_session(session_id)
            message = "Sesión pausada" if success else "No se pudo pausar la sesión"

        elif action == "resume":
            success = await connection_manager.resume_session(session_id)
            message = "Sesión reanudada" if success else "No se pudo reanudar la sesión"

        elif action == "stop":
            success = await connection_manager.stop_session(session_id)
            message = "Sesión detenida" if success else "No se pudo detener la sesión"

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )

        return {
            "success": True,
            "message": message,
            "session_id": session_id,
            "action": action,
            "new_status": session.status.value,
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
                         summary="Eliminar Sesión de Streaming")
async def delete_session(
        session_id: str,
        force: Optional[bool] = False,
        request_id: str = Depends(log_request_info)
):
    """Elimina una sesión de streaming y limpia recursos"""

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
            await connection_manager.stop_session(session_id)
            await real_time_video_service.stop_processing(session_id)

        # Limpiar archivos temporales si existen
        if session.video_path:
            try:
                from services.file_service import file_service
                file_service.cleanup_temp_file(session.video_path)
            except Exception as e:
                logger.warning(f"⚠️ Error limpiando archivo {session.video_path}: {str(e)}")

        # Desconectar sesión
        await connection_manager.disconnect(session_id)

        logger.info(f"🗑️ Sesión eliminada: {session_id}")

        return {
            "success": True,
            "message": f"Sesión {session_id} eliminada exitosamente",
            "session_id": session_id,
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
                      summary="Estado del Sistema de Streaming")
async def streaming_health_check():
    """Verifica el estado del sistema de streaming"""

    try:
        # Obtener métricas del sistema
        system_metrics = connection_manager.metrics.get_stats()
        sessions_info = connection_manager.get_active_sessions_info()

        # Verificar estado de servicios dependientes
        from models.model_manager import model_manager
        models_status = model_manager.get_model_info()

        # Calcular estado general
        total_sessions = sessions_info["total_sessions"]
        active_connections = sessions_info["active_connections"]

        health_status = "healthy"
        if total_sessions > 15:  # Cerca del límite máximo
            health_status = "warning"
        elif system_metrics["websocket_errors"] > 10:
            health_status = "degraded"

        return {
            "status": health_status,
            "timestamp": time.time(),
            "service": "CARID Streaming Service",
            "version": settings.app_version,
            "connections": {
                "total_sessions": total_sessions,
                "active_connections": active_connections,
                "max_sessions": 20
            },
            "models": {
                "loaded": models_status["models_loaded"],
                "device": models_status["device"],
                "plate_detector": models_status["plate_detector_loaded"],
                "char_recognizer": models_status["char_recognizer_loaded"]
            },
            "system_metrics": system_metrics,
            "capabilities": {
                "websocket_streaming": True,
                "real_time_processing": True,
                "multi_session": True,
                "pause_resume": True,
                "frame_visualization": True
            }
        }

    except Exception as e:
        logger.error(f"❌ Error en health check de streaming: {str(e)}")
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }


@streaming_router.get("/config",
                      summary="Configuración del Sistema de Streaming")
async def get_streaming_config():
    """Obtiene la configuración actual del sistema de streaming"""

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
                    "max_concurrent_sessions": 10
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


@streaming_router.get("/stats",
                      summary="Estadísticas del Sistema de Streaming")
async def get_streaming_stats(
        include_detailed: Optional[bool] = False
):
    """Obtiene estadísticas detalladas del sistema de streaming"""

    try:
        # Estadísticas básicas
        sessions_info = connection_manager.get_active_sessions_info()
        system_metrics = connection_manager.metrics.get_stats()

        stats = {
            "overview": {
                "total_sessions": sessions_info["total_sessions"],
                "active_connections": sessions_info["active_connections"],
                "system_uptime": system_metrics["total_time"]
            },
            "performance": {
                "messages_per_second": system_metrics["messages_per_second"],
                "bandwidth_mbps": system_metrics["bandwidth_mbps"],
                "websocket_errors": system_metrics["websocket_errors"]
            },
            "processing": {
                "frames_sent": system_metrics["frames_sent"],
                "avg_processing_time": system_metrics["avg_processing_time"]
            }
        }

        # Incluir detalles si se solicita
        if include_detailed:
            stats["detailed"] = {
                "sessions": sessions_info["sessions"],
                "full_metrics": system_metrics
            }

        return {
            "success": True,
            "data": stats,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo estadísticas: {str(e)}"
        )


@streaming_router.post("/sessions/{session_id}/send-message",
                       summary="Enviar Mensaje a Sesión")
async def send_message_to_session(
        session_id: str,
        message_type: str = Form(..., description="Tipo de mensaje"),
        message_data: Optional[str] = Form(None, description="Datos del mensaje (JSON)"),
        request_id: str = Depends(log_request_info)
):
    """Envía un mensaje personalizado a una sesión específica"""

    try:
        session = connection_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sesión {session_id} no encontrada"
            )

        # Preparar mensaje
        message = {
            "type": message_type,
            "timestamp": time.time()
        }

        # Agregar datos si se proporcionan
        if message_data:
            try:
                import json
                parsed_data = json.loads(message_data)
                message["data"] = parsed_data
            except json.JSONDecodeError:
                message["data"] = message_data

        # Enviar mensaje
        success = await connection_manager.send_message(session_id, message)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudo enviar el mensaje"
            )

        return {
            "success": True,
            "message": "Mensaje enviado exitosamente",
            "session_id": session_id,
            "message_type": message_type,
            "timestamp": time.time()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error enviando mensaje a {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error enviando mensaje: {str(e)}"
        )


@streaming_router.get("/sessions/{session_id}/download-results",
                      summary="Descargar Resultados de Sesión")
async def download_session_results(
        session_id: str,
        format: Optional[str] = "json",
        include_detections: Optional[bool] = True,
        include_frames: Optional[bool] = False
):
    """Descarga los resultados de una sesión en formato especificado"""

    try:
        session = connection_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sesión {session_id} no encontrada"
            )

        # Preparar datos de resultado
        results = {
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
            }
        }

        # Incluir detecciones si se solicita
        if include_detections:
            results["detections"] = session.detections
            results["unique_plates"] = session.unique_plates
            results["best_detection"] = session.best_detection

        # Generar estadísticas adicionales
        if session.detections:
            confidences = [d["overall_confidence"] for d in session.detections]
            results["statistics"] = {
                "avg_confidence": sum(confidences) / len(confidences),
                "max_confidence": max(confidences),
                "min_confidence": min(confidences),
                "valid_plates": len([d for d in session.detections if d.get("is_valid_plate", False)])
            }

        # Retornar según formato solicitado
        if format.lower() == "json":
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=results,
                headers={
                    "Content-Disposition": f"attachment; filename=carid_results_{session_id}.json"
                }
            )
        elif format.lower() == "csv":
            # Convertir a CSV
            import pandas as pd
            from fastapi.responses import StreamingResponse
            import io

            if session.detections:
                df = pd.DataFrame(session.detections)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                return StreamingResponse(
                    io.BytesIO(csv_buffer.getvalue().encode()),
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename=carid_detections_{session_id}.csv"
                    }
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No hay detecciones para exportar a CSV"
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


@streaming_router.post("/sessions/cleanup",
                       summary="Limpiar Sesiones Inactivas")
async def cleanup_inactive_sessions(
        max_age_minutes: Optional[int] = 30,
        force: Optional[bool] = False,
        request_id: str = Depends(log_request_info)
):
    """Limpia sesiones inactivas manualmente"""

    try:
        current_time = time.time()
        max_age_seconds = max_age_minutes * 60

        sessions_to_cleanup = []
        for session_id, session in connection_manager.sessions.items():
            # Determinar si la sesión debe limpiarse
            should_cleanup = False

            if force:
                should_cleanup = True
            elif session.last_activity:
                age = current_time - session.last_activity
                if age > max_age_seconds:
                    should_cleanup = True
            elif session.status in [StreamingStatus.ERROR, StreamingStatus.COMPLETED]:
                if session.start_time:
                    age = current_time - session.start_time
                    if age > max_age_seconds:
                        should_cleanup = True

            if should_cleanup:
                sessions_to_cleanup.append(session_id)

        # Limpiar sesiones identificadas
        cleaned_count = 0
        for session_id in sessions_to_cleanup:
            try:
                await connection_manager.disconnect(session_id)
                cleaned_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Error limpiando sesión {session_id}: {str(e)}")

        return {
            "success": True,
            "message": f"Limpieza completada: {cleaned_count} sesiones eliminadas",
            "sessions_cleaned": cleaned_count,
            "sessions_checked": len(connection_manager.sessions) + cleaned_count,
            "criteria": {
                "max_age_minutes": max_age_minutes,
                "force": force
            },
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"❌ Error en limpieza de sesiones: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en limpieza: {str(e)}"
        )


@streaming_router.websocket("/test/{session_id}")
async def websocket_test_connection(websocket: WebSocket, session_id: str):
    """WebSocket de prueba para verificar conectividad"""

    try:
        await websocket.accept()

        # Enviar mensaje de bienvenida
        await websocket.send_text(json.dumps({
            "type": "test_connection",
            "message": "Conexión WebSocket de prueba establecida",
            "session_id": session_id,
            "timestamp": time.time()
        }))

        # Loop de prueba simple
        counter = 0
        while True:
            try:
                # Enviar ping cada 5 segundos
                await asyncio.sleep(5)
                counter += 1

                await websocket.send_text(json.dumps({
                    "type": "test_ping",
                    "counter": counter,
                    "message": f"Ping #{counter}",
                    "timestamp": time.time()
                }))

                # Recibir respuesta si la hay
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    message = json.loads(data)

                    await websocket.send_text(json.dumps({
                        "type": "test_echo",
                        "received": message,
                        "timestamp": time.time()
                    }))
                except asyncio.TimeoutError:
                    pass  # No hay mensaje del cliente, continuar

            except WebSocketDisconnect:
                logger.info(f"🔌 Test WebSocket {session_id} desconectado")
                break

    except Exception as e:
        logger.error(f"❌ Error en test WebSocket {session_id}: {str(e)}")
    finally:
        logger.info(f"🧪 Test WebSocket {session_id} finalizado")


# Middleware personalizado para logging de WebSocket
@streaming_router.middleware("http")
async def log_streaming_requests(request: Request, call_next):
    """Middleware para logging específico de streaming"""

    start_time = time.time()

    # Log de request entrante
    if request.url.path.startswith("/api/v1/streaming"):
        logger.info(f"🌐 Streaming request: {request.method} {request.url.path}")

    response = await call_next(request)

    # Log de response
    process_time = time.time() - start_time
    if request.url.path.startswith("/api/v1/streaming"):
        logger.info(f"✅ Streaming response: {response.status_code} in {process_time:.3f}s")

    return response_code=status.HTTP_404_NOT_FOUND,
    detail = "Sesión WebSocket no encontrada. Conecta primero vía WebSocket."

)

if not session.websocket:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Sesión WebSocket no está activa"
    )

# Validar que los modelos estén cargados
if not models.is_loaded:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Los modelos ALPR no están cargados"
    )

# Validar archivo
if not file.filename:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Nombre de archivo requerido"
    )

# Verificar extensión de video
video_extensions = settings.video_extensions_list
file_extension = file.filename.split('.')[-1].lower()

if file_extension not in video_extensions:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Formato no soportado. Extensiones válidas: {', '.join(video_extensions)}"
    )

# Guardar archivo usando el servicio existente
from services.file_service import file_service

file_path, file_info = await file_service.save_upload_file(file, "streaming_")

# Actualizar estado de la sesión
connection_manager.update_session_status(session_id, StreamingStatus.INITIALIZING)

# Configurar parámetros de procesamiento
processing_params = {
    "session_id": session_id,
    "confidence_threshold": confidence_threshold,
    "iou_threshold": iou_threshold,
    "frame_skip": frame_skip,
    "max_duration": max_duration,
    "enhance_processing": enhance_processing,
    "save_best_frames": save_best_frames,
    "real_time_mode": True
}

# Actualizar información en la sesión
session.video_path = file_path
session.file_info = file_info
session.processing_params = processing_params

# Enviar confirmación inmediata
await connection_manager.send_message(session_id, {
    "type": "processing_accepted",
    "data": {
        "file_info": file_info,
        "processing_params": processing_params,
        "status": "initializing"
    }
})

# Iniciar procesamiento en background (no bloquear la respuesta)
asyncio.create_task(
    real_time_video_service.process_video_streaming(
        file_path, file_info, processing_params
    )
)

logger.success(f"✅ Procesamiento iniciado para {session_id}: {file_info['filename']}")

return {
    "success": True,
    "message": "Procesamiento iniciado exitosamente",
    "session_id": session_id,
    "file_info": {
        "filename": file_info["filename"],
        "size_mb": file_info["size_mb"],
        "file_type": file_info.get("file_type", "video")
    },
    "processing_params": processing_params,
    "websocket_status": "connected",
    "timestamp": time.time()
}

except HTTPException:
raise
except Exception as e:
logger.error(f"❌ Error iniciando procesamiento {session_id}: {str(e)}")

# Notificar error vía WebSocket si es posible
try:
    await connection_manager.send_message(session_id, {
        "type": "processing_error",
        "error": str(e)
    })
except:
    pass

raise HTTPException(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    detail=f"Error interno: {str(e)}"
)


@streaming_router.get("/sessions",
                      summary="Obtener Sesiones Activas",
                      description="Retorna información de todas las sesiones de streaming activas")
async def get_active_sessions(
        include_metrics: Optional[bool] = False,
        request_id: str = Depends(log_request_info)
):
    """Obtiene información de sesiones activas de streaming"""

    try:
        sessions_info = connection_manager.get_active_sessions_info()

        if not include_metrics:
            # Remover métricas del sistema si no se solicitan
            sessions_info.pop("system_metrics", None)

        return {
            "success": True,
            "data": sessions_info,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo sesiones: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo información de sesiones: {str(e)}"
        )


@streaming_router.get("/sessions/{session_id}",
                      summary="Obtener Información de Sesión Específica")
async def get_session_info(
        session_id: str,
        include_detections: Optional[bool] = False,
        request_id: str = Depends(log_request_info)
):
    """Obtiene información detallada de una sesión específica"""

    try:
        session = connection_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status