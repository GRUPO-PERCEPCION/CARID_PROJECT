"""
🎬 NUEVO Sistema de Streaming WebSocket - Completamente reescrito
Endpoints limpios y organizados para streaming en tiempo real
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi import Depends, HTTPException, status, Query
from typing import Optional, Dict, Any, List
import asyncio
import json
import time
import uuid
from loguru import logger

# Importaciones internas
from config.settings import settings
from models.model_manager import model_manager
from services.file_service import file_service
from api.dependencies import get_model_manager, log_request_info

# 🚀 ROUTER PRINCIPAL
streaming_router = APIRouter(
    prefix="/api/v1/streaming",
    tags=["🎬 Video Streaming"]
)

# 📊 ESTADO GLOBAL DE SESIONES (Simplificado)
active_sessions: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}
cleanup_task_started = False


class StreamingSession:
    """Clase simple para manejar sesiones de streaming"""

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.created_at = time.time()
        self.last_activity = time.time()
        self.status = "connected"
        self.video_path = None
        self.is_processing = False

    def update_activity(self):
        self.last_activity = time.time()

    async def send_message(self, message: Dict[str, Any]):
        """Envía mensaje al cliente WebSocket"""
        try:
            await self.websocket.send_text(json.dumps(message))
            self.update_activity()
            return True
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje a {self.session_id}: {str(e)}")
            return False

    async def disconnect(self):
        """Desconecta la sesión"""
        try:
            if self.websocket:
                await self.websocket.close()
        except:
            pass
        self.status = "disconnected"


# 🔧 FUNCIONES AUXILIARES

def get_session(session_id: str) -> Optional[StreamingSession]:
    """Obtiene una sesión activa"""
    return active_sessions.get(session_id)

def cleanup_session(session_id: str):
    """Limpia una sesión"""
    if session_id in active_sessions:
        del active_sessions[session_id]
    if session_id in websocket_connections:
        del websocket_connections[session_id]

async def broadcast_to_session(session_id: str, message_type: str, data: Any):
    """Envía mensaje a una sesión específica"""
    session = get_session(session_id)
    if session:
        message = {
            "type": message_type,
            "data": data,
            "timestamp": time.time()
        }
        return await session.send_message(message)
    return False

async def start_cleanup_task_if_needed():
    """Inicia la tarea de limpieza si no está iniciada"""
    global cleanup_task_started

    if not cleanup_task_started:
        try:
            # Crear la tarea de limpieza
            asyncio.create_task(cleanup_inactive_sessions())
            cleanup_task_started = True
            logger.info("🧹 Tarea de limpieza iniciada")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo iniciar tarea de limpieza: {str(e)}")


# 🔌 ENDPOINTS WEBSOCKET

@streaming_router.websocket("/ws/{session_id}")
async def streaming_websocket(websocket: WebSocket, session_id: str):
    """
    🎯 ENDPOINT PRINCIPAL DE STREAMING

    Conecta cliente para streaming en tiempo real
    URL: ws://localhost:8000/api/v1/streaming/ws/{session_id}
    """

    try:
        # Iniciar cleanup si es necesario
        await start_cleanup_task_if_needed()

        # Aceptar conexión
        await websocket.accept()
        logger.info(f"🔌 WebSocket conectado: {session_id}")

        # Crear sesión
        session = StreamingSession(session_id, websocket)
        active_sessions[session_id] = session
        websocket_connections[session_id] = websocket

        # Enviar confirmación de conexión
        await session.send_message({
            "type": "connection_established",
            "session_id": session_id,
            "message": "Conexión WebSocket establecida correctamente",
            "server_info": {
                "service": "CARID Streaming",
                "version": settings.app_version,
                "max_file_size_mb": settings.max_file_size,
                "supported_formats": settings.video_extensions_list
            },
            "streaming_config": {
                "frame_quality": settings.streaming_frame_quality,
                "max_duration": settings.max_video_duration,
                "adaptive_quality": settings.streaming_adaptive_quality
            }
        })

        # 🔄 LOOP PRINCIPAL DE MENSAJES
        while True:
            try:
                # Recibir mensaje con timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Parsear mensaje
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    await session.send_message({
                        "type": "error",
                        "error": "Formato JSON inválido"
                    })
                    continue

                # Procesar mensaje
                await handle_websocket_message(session, message)

            except asyncio.TimeoutError:
                # Enviar ping para mantener conexión
                await session.send_message({
                    "type": "ping",
                    "timestamp": time.time()
                })

            except WebSocketDisconnect:
                logger.info(f"🔌 Cliente {session_id} desconectado")
                break

            except Exception as e:
                logger.error(f"❌ Error en WebSocket {session_id}: {str(e)}")
                await session.send_message({
                    "type": "error",
                    "error": f"Error interno: {str(e)}"
                })
                break

    except Exception as e:
        logger.error(f"❌ Error estableciendo WebSocket {session_id}: {str(e)}")

    finally:
        # Limpiar sesión
        cleanup_session(session_id)
        logger.info(f"🧹 Sesión {session_id} limpiada")


@streaming_router.websocket("/test/{session_id}")
async def test_websocket(websocket: WebSocket, session_id: str):
    """
    🧪 ENDPOINT DE PRUEBA SIMPLE

    Para verificar conectividad básica
    URL: ws://localhost:8000/api/v1/streaming/test/{session_id}
    """

    try:
        await websocket.accept()
        logger.info(f"🧪 Test WebSocket conectado: {session_id}")

        # Mensaje de bienvenida
        await websocket.send_text(json.dumps({
            "type": "test_connected",
            "session_id": session_id,
            "message": "Test WebSocket funcionando correctamente",
            "timestamp": time.time()
        }))

        # Loop simple de echo
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"📥 Test recibido: {data}")

                # Echo response
                await websocket.send_text(json.dumps({
                    "type": "echo",
                    "original": data,
                    "session_id": session_id,
                    "timestamp": time.time()
                }))

            except WebSocketDisconnect:
                logger.info(f"🧪 Test {session_id} desconectado")
                break
            except Exception as e:
                logger.error(f"❌ Error en test: {str(e)}")
                break

    except Exception as e:
        logger.error(f"❌ Error en test WebSocket: {str(e)}")
    finally:
        logger.info(f"🧪 Test {session_id} finalizado")


# 🎮 MANEJADOR DE MENSAJES

async def handle_websocket_message(session: StreamingSession, message: Dict[str, Any]):
    """Maneja mensajes recibidos del cliente - VERSIÓN CORREGIDA"""

    message_type = message.get("type", "")
    data = message.get("data", {})

    try:
        logger.info(f"🎯 Procesando mensaje: {message_type} para sesión {session.session_id}")

        if message_type == "ping":
            # Responder pong
            await session.send_message({
                "type": "pong",
                "timestamp": time.time()
            })

        elif message_type == "get_status":
            # Enviar estado de la sesión
            await session.send_message({
                "type": "status",
                "data": {
                    "session_id": session.session_id,
                    "status": session.status,
                    "created_at": session.created_at,
                    "uptime": time.time() - session.created_at,
                    "is_processing": session.is_processing,
                    "video_loaded": session.video_path is not None
                }
            })

        elif message_type == "start_processing":
            """🔧 HANDLER CLAVE - INICIAR PROCESAMIENTO"""
            logger.info(f"🎬 Iniciando procesamiento para sesión {session.session_id}")

            if not session.video_path:
                await session.send_message({
                    "type": "error",
                    "error": "No hay video cargado para procesar"
                })
                return

            # Marcar como procesando
            session.is_processing = True
            await session.send_message({
                "type": "streaming_started",
                "data": {
                    "message": "Procesamiento de streaming iniciado",
                    "video_path": session.video_path,
                    "session_id": session.session_id
                }
            })

            # 🚀 INICIAR PROCESAMIENTO DE VIDEO EN BACKGROUND
            try:
                from services.streaming_service import streaming_service
                success = await streaming_service.start_video_streaming(
                    session.session_id,
                    session.video_path,
                    getattr(session, 'file_info', {}),
                    data.get('options', {})
                )

                if not success:
                    await session.send_message({
                        "type": "streaming_error",
                        "error": "No se pudo iniciar el procesamiento de video"
                    })
                    session.is_processing = False

            except Exception as e:
                logger.error(f"❌ Error iniciando procesamiento: {str(e)}")
                await session.send_message({
                    "type": "streaming_error",
                    "error": f"Error iniciando procesamiento: {str(e)}"
                })
                session.is_processing = False

        elif message_type == "stop_processing":
            # Detener procesamiento
            session.is_processing = False
            await session.send_message({
                "type": "processing_stopped",
                "message": "Procesamiento detenido"
            })

            # Detener en el servicio de streaming
            try:
                from services.streaming_service import streaming_service
                await streaming_service.stop_streaming(session.session_id)
            except Exception as e:
                logger.warning(f"⚠️ Error deteniendo streaming: {str(e)}")

        else:
            # Mensaje no reconocido
            logger.warning(f"❓ Tipo de mensaje no soportado: {message_type}")
            await session.send_message({
                "type": "error",
                "error": f"Tipo de mensaje no soportado: {message_type}",
                "supported_types": [
                    "ping", "get_status", "start_processing", "stop_processing"
                ]
            })

    except Exception as e:
        logger.error(f"❌ Error procesando mensaje {message_type}: {str(e)}")
        await session.send_message({
            "type": "error",
            "error": f"Error procesando mensaje: {str(e)}"
        })


# 📡 ENDPOINTS REST

@streaming_router.post("/upload")
async def upload_video_for_streaming(
        session_id: str = Form(..., description="ID de sesión WebSocket"),
        file: UploadFile = File(..., description="Video a procesar"),
        confidence_threshold: Optional[float] = Form(0.3, description="Umbral de confianza"),
        frame_skip: Optional[int] = Form(2, description="Salto de frames"),
        max_duration: Optional[int] = Form(600, description="Duración máxima"),
        request_id: str = Depends(log_request_info),
        models=Depends(get_model_manager)
):
    """📤 UPLOAD CORREGIDO CON AUTO-START"""

    try:
        # Verificar que la sesión existe
        session = get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "SESSION_NOT_FOUND",
                    "message": f"Sesión WebSocket {session_id} no encontrada",
                    "hint": "Conecta primero vía WebSocket"
                }
            )

        # Validar archivo (código existente)...
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nombre de archivo requerido"
            )

        # Verificar extensión
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.video_extensions_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "UNSUPPORTED_FORMAT",
                    "message": f"Formato no soportado: {file_extension}",
                    "supported": settings.video_extensions_list
                }
            )

        # Guardar archivo
        file_path, file_info = await file_service.save_upload_file(file, "streaming_")
        session.video_path = file_path

        # 🔧 GUARDAR INFO DEL ARCHIVO EN LA SESIÓN
        session.file_info = file_info

        # Notificar al cliente vía WebSocket
        await session.send_message({
            "type": "video_uploaded",
            "data": {
                "filename": file_info["filename"],
                "size_mb": file_info["size_mb"],
                "file_type": file_info["file_type"],
                "dimensions": file_info.get("dimensions"),
                "ready_for_processing": True
            }
        })

        logger.info(f"📤 Video subido para {session_id}: {file_info['filename']}")

        # 🚀 AUTO-INICIAR PROCESAMIENTO DESPUÉS DE 2 SEGUNDOS
        async def auto_start_processing():
            await asyncio.sleep(2)
            try:
                logger.info(f"🎬 Auto-iniciando procesamiento para {session_id}")

                # Configurar parámetros
                processing_params = {
                    "confidence_threshold": confidence_threshold,
                    "frame_skip": frame_skip,
                    "max_duration": max_duration
                }

                # Iniciar streaming
                from services.streaming_service import streaming_service
                success = await streaming_service.start_video_streaming(
                    session_id,
                    file_path,
                    file_info,
                    processing_params
                )

                if success:
                    logger.info(f"✅ Streaming auto-iniciado para {session_id}")
                    session.is_processing = True
                else:
                    logger.error(f"❌ Error auto-iniciando streaming para {session_id}")
                    await session.send_message({
                        "type": "streaming_error",
                        "error": "No se pudo auto-iniciar el procesamiento"
                    })

            except Exception as e:
                logger.error(f"❌ Error en auto-start: {str(e)}")
                await session.send_message({
                    "type": "streaming_error",
                    "error": f"Error auto-iniciando: {str(e)}"
                })

        # Ejecutar auto-start en background
        asyncio.create_task(auto_start_processing())

        return {
            "success": True,
            "message": "Video subido correctamente - Procesamiento iniciará automáticamente",
            "session_id": session_id,
            "file_info": file_info,
            "processing_will_start": True,
            "next_steps": [
                "El video está listo para procesamiento",
                "El streaming iniciará automáticamente en 2 segundos",
                "Mantén la conexión WebSocket abierta para recibir actualizaciones"
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error subiendo video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )


@streaming_router.get("/sessions")
async def list_active_sessions():
    """📋 LISTAR SESIONES ACTIVAS"""

    try:
        session_list = []

        for session_id, session in active_sessions.items():
            session_info = {
                "session_id": session_id,
                "status": session.status,
                "created_at": session.created_at,
                "uptime": time.time() - session.created_at,
                "last_activity": session.last_activity,
                "is_processing": session.is_processing,
                "has_video": session.video_path is not None
            }
            session_list.append(session_info)

        return {
            "success": True,
            "total_sessions": len(session_list),
            "sessions": session_list,
            "server_capacity": {
                "max_connections": settings.max_websocket_connections,
                "current_connections": len(active_sessions),
                "available_slots": settings.max_websocket_connections - len(active_sessions)
            }
        }

    except Exception as e:
        logger.error(f"❌ Error listando sesiones: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo sesiones: {str(e)}"
        )


@streaming_router.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """📄 INFO DE SESIÓN ESPECÍFICA"""

    session = get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sesión {session_id} no encontrada"
        )

    return {
        "success": True,
        "session": {
            "session_id": session.session_id,
            "status": session.status,
            "created_at": session.created_at,
            "uptime": time.time() - session.created_at,
            "last_activity": session.last_activity,
            "is_processing": session.is_processing,
            "video_path": session.video_path,
            "has_video": session.video_path is not None
        }
    }


@streaming_router.delete("/sessions/{session_id}")
async def disconnect_session(session_id: str):
    """🗑️ DESCONECTAR SESIÓN"""

    session = get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sesión {session_id} no encontrada"
        )

    # Desconectar
    await session.disconnect()
    cleanup_session(session_id)

    return {
        "success": True,
        "message": f"Sesión {session_id} desconectada correctamente"
    }


@streaming_router.get("/health")
async def streaming_health():
    """🏥 HEALTH CHECK DE STREAMING"""

    try:
        # Estado de modelos
        models_info = model_manager.get_model_info() if model_manager.is_loaded else {"models_loaded": False}

        # Estado de sesiones
        active_count = len(active_sessions)
        max_count = settings.max_websocket_connections

        health_status = "healthy"
        issues = []

        if not models_info.get("models_loaded", False):
            health_status = "degraded"
            issues.append("Modelos ALPR no cargados")

        if active_count > max_count * 0.8:
            health_status = "warning"
            issues.append("Acercándose al límite de sesiones")

        return {
            "status": health_status,
            "timestamp": time.time(),
            "service": "CARID Streaming Service",
            "version": settings.app_version,
            "issues": issues,
            "sessions": {
                "active": active_count,
                "max": max_count,
                "capacity_usage": (active_count / max_count) * 100
            },
            "models": {
                "loaded": models_info.get("models_loaded", False),
                "device": models_info.get("device", "unknown")
            },
            "capabilities": {
                "websocket_streaming": True,
                "real_time_processing": True,
                "video_upload": True,
                "session_management": True
            }
        }

    except Exception as e:
        logger.error(f"❌ Error en health check: {str(e)}")
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }


@streaming_router.get("/test-connection")
async def test_connection():
    """🧪 PROBAR CONECTIVIDAD"""

    return {
        "success": True,
        "message": "Servicio de streaming funcionando correctamente",
        "endpoints": {
            "websocket_main": "/api/v1/streaming/ws/{session_id}",
            "websocket_test": "/api/v1/streaming/test/{session_id}",
            "upload_video": "/api/v1/streaming/upload",
            "list_sessions": "/api/v1/streaming/sessions",
            "health_check": "/api/v1/streaming/health"
        },
        "example_usage": {
            "step_1": "Conectar WebSocket: ws://localhost:8000/api/v1/streaming/ws/mi_sesion",
            "step_2": "Subir video: POST /api/v1/streaming/upload",
            "step_3": "Enviar mensaje: {'type': 'start_processing'}",
            "step_4": "Recibir frames procesados en tiempo real"
        },
        "timestamp": time.time()
    }


# 🔄 CLEANUP TASK (Corregido - sin create_task en import)
async def cleanup_inactive_sessions():
    """Limpia sesiones inactivas cada 5 minutos"""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutos

            current_time = time.time()
            inactive_sessions = []

            for session_id, session in active_sessions.items():
                # Sesiones inactivas por más de 30 minutos
                if current_time - session.last_activity > 1800:
                    inactive_sessions.append(session_id)

            # Limpiar sesiones inactivas
            for session_id in inactive_sessions:
                logger.info(f"🧹 Limpiando sesión inactiva: {session_id}")
                session = active_sessions.get(session_id)
                if session:
                    await session.disconnect()
                cleanup_session(session_id)

        except Exception as e:
            logger.error(f"❌ Error en cleanup: {str(e)}")

# 📊 ESTADÍSTICAS FINALES
logger.info("🎬 Nuevo sistema de streaming inicializado")
logger.info("🔌 Endpoints WebSocket disponibles:")
logger.info("   • Principal: /api/v1/streaming/ws/{session_id}")
logger.info("   • Prueba: /api/v1/streaming/test/{session_id}")
logger.info("📡 Endpoints REST disponibles:")
logger.info("   • Upload: POST /api/v1/streaming/upload")
logger.info("   • Sessions: GET /api/v1/streaming/sessions")
logger.info("   • Health: GET /api/v1/streaming/health")