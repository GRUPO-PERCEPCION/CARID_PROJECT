# api/routes/streaming.py - CORRECCIÓN DEL TIMING
"""
🎬 Sistema de Streaming WebSocket - CORRECCIÓN TIMING
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi import HTTPException, status
from typing import Dict, Any, List, Optional
import asyncio
import json
import time
import os
from loguru import logger

# Importaciones internas básicas
from config.settings import settings
from models.model_manager import model_manager
from services.file_service import file_service

# 🚀 ROUTER PRINCIPAL
streaming_router = APIRouter(
    prefix="/api/v1/streaming",
    tags=["🎬 Video Streaming"]
)

# 📊 ESTADO GLOBAL DE SESIONES
active_sessions: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}


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

        # Atributos para streaming_service
        self.file_info = {}
        self.video_info = {}
        self.processing_params = {}
        self.total_frames = 0
        self.start_time = None
        self.processed_frames = 0
        self.frames_with_detections = 0
        self.total_detection_count = 0
        self.unique_plates = {}
        self.best_detection = None
        self.processing_speed = 0.0
        self.should_stop = False
        self.is_paused = False
        self.current_frame = 0

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


def get_session(session_id: str) -> Optional[StreamingSession]:
    """Obtiene una sesión activa"""
    logger.debug(f"🔍 Buscando sesión: {session_id}")
    logger.debug(f"🔍 Sesiones activas: {list(active_sessions.keys())}")

    session = active_sessions.get(session_id)
    if session:
        logger.debug(f"✅ Sesión encontrada: {session_id}")
        return session
    else:
        logger.warning(f"❌ Sesión NO encontrada: {session_id}")
        return None


def cleanup_session(session_id: str):
    """Limpia una sesión"""
    if session_id in active_sessions:
        del active_sessions[session_id]
    if session_id in websocket_connections:
        del websocket_connections[session_id]


# 🔌 WEBSOCKET ENDPOINT
@streaming_router.websocket("/ws/{session_id}")
async def streaming_websocket(websocket: WebSocket, session_id: str):
    """WebSocket principal para streaming"""

    try:
        await websocket.accept()
        logger.info(f"🔌 WebSocket conectado: {session_id}")

        # Crear sesión
        session = StreamingSession(session_id, websocket)
        active_sessions[session_id] = session
        websocket_connections[session_id] = websocket

        logger.info(f"📝 Sesión registrada: {session_id}")
        logger.debug(f"📝 Sesiones activas ahora: {list(active_sessions.keys())}")

        # Enviar confirmación
        await session.send_message({
            "type": "connection_established",
            "session_id": session_id,
            "message": "Conexión WebSocket establecida correctamente",
            "server_info": {
                "service": "CARID Streaming",
                "version": settings.app_version,
                "max_file_size_mb": settings.max_file_size
            }
        })

        # Loop de mensajes
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                try:
                    message = json.loads(data)
                    await handle_websocket_message(session, message)
                except json.JSONDecodeError:
                    await session.send_message({
                        "type": "error",
                        "error": "Formato JSON inválido"
                    })

            except asyncio.TimeoutError:
                await session.send_message({"type": "ping", "timestamp": time.time()})
            except WebSocketDisconnect:
                logger.info(f"🔌 Cliente {session_id} desconectado")
                break
            except Exception as e:
                logger.error(f"❌ Error en WebSocket {session_id}: {str(e)}")
                break

    except Exception as e:
        logger.error(f"❌ Error estableciendo WebSocket {session_id}: {str(e)}")
    finally:
        cleanup_session(session_id)
        logger.info(f"🧹 Sesión {session_id} limpiada")


async def handle_websocket_message(session: StreamingSession, message: Dict[str, Any]):
    """Maneja mensajes WebSocket"""

    message_type = message.get("type", "")
    data = message.get("data", {})

    try:
        logger.info(f"🎯 Mensaje recibido: {message_type} para {session.session_id}")

        if message_type == "ping":
            await session.send_message({"type": "pong", "timestamp": time.time()})

        elif message_type == "get_status":
            await session.send_message({
                "type": "status",
                "data": {
                    "session_id": session.session_id,
                    "status": session.status,
                    "is_processing": session.is_processing,
                    "video_loaded": session.video_path is not None
                }
            })

        elif message_type == "pause_processing":
            session.is_paused = True
            await session.send_message({"type": "processing_paused"})

        elif message_type == "resume_processing":
            session.is_paused = False
            await session.send_message({"type": "processing_resumed"})

        elif message_type == "stop_processing":
            session.should_stop = True
            session.is_processing = False
            await session.send_message({"type": "processing_stopped"})

    except Exception as e:
        logger.error(f"❌ Error procesando mensaje {message_type}: {str(e)}")


# 📡 UPLOAD ENDPOINT CORREGIDO
@streaming_router.post("/upload")
async def upload_video_for_streaming(
        session_id: str = Form(...),
        file: UploadFile = File(...),
        confidence_threshold: Optional[float] = Form(0.3),
        frame_skip: Optional[int] = Form(2),
        max_duration: Optional[int] = Form(600)
):
    """Upload de video con timing corregido"""

    try:
        logger.info(f"📤 [UPLOAD] Upload para sesión: {session_id}")

        # 🔧 VERIFICAR SESIÓN CON RETRY
        session = None
        max_retries = 5
        for attempt in range(max_retries):
            session = get_session(session_id)
            if session:
                break
            logger.warning(f"⚠️ [UPLOAD] Intento {attempt + 1}/{max_retries} - Sesión no encontrada, esperando...")
            await asyncio.sleep(0.5)  # Esperar 500ms antes del siguiente intento

        if not session:
            logger.error(f"❌ [UPLOAD] Sesión no encontrada después de {max_retries} intentos: {session_id}")
            logger.debug(f"❌ [UPLOAD] Sesiones disponibles: {list(active_sessions.keys())}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "SESSION_NOT_FOUND",
                    "message": f"Sesión {session_id} no encontrada después de {max_retries} intentos",
                    "available_sessions": list(active_sessions.keys())
                }
            )

        logger.info(f"✅ [UPLOAD] Sesión encontrada: {session_id}")

        # Verificar modelos
        if not model_manager.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Modelos no cargados"
            )

        # Validar archivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="Archivo requerido")

        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.video_extensions_list:
            raise HTTPException(
                status_code=400,
                detail=f"Formato {file_extension} no soportado"
            )

        # Guardar archivo
        logger.info(f"💾 [UPLOAD] Guardando archivo: {file.filename}")
        file_path, file_info = await file_service.save_upload_file(file, "streaming_")
        session.video_path = file_path
        session.file_info = file_info

        logger.info(f"✅ [UPLOAD] Archivo guardado: {file_path}")

        # Notificar upload
        await session.send_message({
            "type": "video_uploaded",
            "data": {
                "filename": file_info["filename"],
                "size_mb": file_info["size_mb"],
                "ready_for_processing": True
            }
        })

        # 🔧 CONFIGURAR PARÁMETROS ANTES DEL AUTO-START
        processing_params = {
            "confidence_threshold": confidence_threshold,
            "frame_skip": frame_skip,
            "max_duration": max_duration
        }
        session.processing_params = processing_params

        logger.info(f"⚙️ [UPLOAD] Parámetros configurados: {processing_params}")

        # 🚀 AUTO-INICIAR PROCESAMIENTO CON DELAY Y VERIFICACIÓN
        async def auto_start():
            try:
                # Esperar un poco más para asegurar que todo esté listo
                await asyncio.sleep(1)

                logger.info(f"🎬 [AUTO-START] Iniciando procesamiento para {session_id}")

                # Verificar que la sesión sigue existiendo
                current_session = get_session(session_id)
                if not current_session:
                    logger.error(f"❌ [AUTO-START] Sesión perdida durante auto-start: {session_id}")
                    return

                # Importar streaming service
                from services.streaming_service import streaming_service

                logger.info(f"🔧 [AUTO-START] Llamando a start_video_streaming...")

                success = await streaming_service.start_video_streaming(
                    session_id, file_path, file_info, processing_params
                )

                if success:
                    current_session.is_processing = True
                    logger.info(f"✅ [AUTO-START] Streaming iniciado exitosamente: {session_id}")
                else:
                    logger.error(f"❌ [AUTO-START] Error iniciando streaming: {session_id}")
                    await current_session.send_message({
                        "type": "streaming_error",
                        "error": "No se pudo iniciar procesamiento"
                    })

            except Exception as e:
                logger.error(f"❌ [AUTO-START] Error en auto-start: {str(e)}")
                logger.exception("Stack trace del auto-start:")

                # Intentar notificar el error
                try:
                    error_session = get_session(session_id)
                    if error_session:
                        await error_session.send_message({
                            "type": "streaming_error",
                            "error": f"Error auto-iniciando: {str(e)}"
                        })
                except Exception as notify_error:
                    logger.error(f"❌ [AUTO-START] Error notificando error: {str(notify_error)}")

        # Ejecutar en background
        asyncio.create_task(auto_start())

        return {
            "success": True,
            "message": "Video subido - procesamiento iniciará automáticamente",
            "session_id": session_id,
            "file_info": file_info,
            "processing_will_start": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [UPLOAD] Error inesperado: {str(e)}")
        logger.exception("Stack trace del upload:")
        raise HTTPException(status_code=500, detail=str(e))


# 📋 OTROS ENDPOINTS BÁSICOS
@streaming_router.get("/sessions")
async def list_sessions():
    """Lista sesiones activas"""
    sessions = []
    for sid, session in active_sessions.items():
        sessions.append({
            "session_id": sid,
            "status": session.status,
            "is_processing": session.is_processing,
            "has_video": session.video_path is not None,
            "uptime": time.time() - session.created_at,
            "last_activity": session.last_activity,
            "created_at": session.created_at
        })

    return {
        "success": True,
        "total_sessions": len(sessions),
        "sessions": sessions,
        "server_capacity": {
            "max_connections": settings.max_websocket_connections,
            "current_connections": len(active_sessions),
            "available_slots": settings.max_websocket_connections - len(active_sessions)
        }
    }


@streaming_router.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Info de sesión específica"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Sesión {session_id} no encontrada")

    return {
        "success": True,
        "session": {
            "session_id": session.session_id,
            "status": session.status,
            "is_processing": session.is_processing,
            "has_video": session.video_path is not None,
            "uptime": time.time() - session.created_at,
            "last_activity": session.last_activity,
            "created_at": session.created_at,
            "video_path": session.video_path,
            "processed_frames": getattr(session, 'processed_frames', 0),
            "total_frames": getattr(session, 'total_frames', 0)
        }
    }


@streaming_router.delete("/sessions/{session_id}")
async def disconnect_session(session_id: str):
    """Desconectar sesión"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Sesión {session_id} no encontrada")

    await session.disconnect()
    cleanup_session(session_id)

    return {
        "success": True,
        "message": f"Sesión {session_id} desconectada"
    }


@streaming_router.get("/health")
async def health_check():
    """Health check básico"""
    active_count = len(active_sessions)
    max_count = settings.max_websocket_connections

    health_status = "healthy"
    issues = []

    if not model_manager.is_loaded:
        health_status = "warning"
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
            "loaded": model_manager.is_loaded,
            "device": settings.device if model_manager.is_loaded else "unknown"
        },
        "capabilities": {
            "websocket_streaming": True,
            "real_time_processing": True,
            "video_upload": True,
            "session_management": True
        }
    }


@streaming_router.get("/test-connection")
async def test_connection():
    """Test de conectividad"""
    return {
        "success": True,
        "message": "Servicio funcionando",
        "timestamp": time.time(),
        "active_sessions": len(active_sessions),
        "session_ids": list(active_sessions.keys())
    }


logger.info("🎬 Streaming router con timing corregido inicializado")
logger.info("🎬 Streaming router con timing corregido inicializado")