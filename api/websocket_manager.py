"""
Sistema WebSocket Manager completo para streaming de video ALPR
"""

import asyncio
import json
import uuid
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from enum import Enum
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor


class StreamingStatus(str, Enum):
    """Estados del streaming de video"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StreamingSession:
    """Datos de una sesión de streaming"""
    session_id: str
    websocket: Optional[WebSocket] = None
    status: StreamingStatus = StreamingStatus.DISCONNECTED

    # Información del video
    video_path: Optional[str] = None
    file_info: Optional[Dict[str, Any]] = None
    video_info: Optional[Dict[str, Any]] = None

    # Progreso de procesamiento
    total_frames: int = 0
    processed_frames: int = 0
    current_frame: int = 0

    # Detecciones y resultados
    detections: List[Dict[str, Any]] = field(default_factory=list)
    unique_plates: Dict[str, Any] = field(default_factory=dict)

    # Métricas de tiempo
    start_time: Optional[float] = None
    last_activity: Optional[float] = None
    processing_speed: float = 0.0

    # Configuración de procesamiento
    processing_params: Dict[str, Any] = field(default_factory=dict)

    # Control de flujo
    is_paused: bool = False
    should_stop: bool = False

    # Métricas adicionales
    frames_with_detections: int = 0
    total_detection_count: int = 0
    best_detection: Optional[Dict[str, Any]] = None


class SecurityManager:
    """Gestor de seguridad para conexiones WebSocket"""

    def __init__(self):
        self.connection_attempts: Dict[str, List[float]] = defaultdict(list)
        self.active_sessions: Dict[str, str] = {}  # session_id -> client_ip
        self.max_sessions_per_ip = 3
        self.rate_limit_window = 60  # segundos
        self.max_total_sessions = 20

    def can_connect(self, client_ip: str, session_id: str) -> tuple[bool, str]:
        """
        Valida si un cliente puede conectarse

        Returns:
            (can_connect, reason)
        """
        current_time = time.time()

        # Verificar límite total de sesiones
        if len(self.active_sessions) >= self.max_total_sessions:
            return False, "Límite máximo de sesiones alcanzado"

        # Verificar sesión duplicada
        if session_id in self.active_sessions:
            return False, "Sesión ya existe"

        # Limpiar intentos antiguos
        self.connection_attempts[client_ip] = [
            t for t in self.connection_attempts[client_ip]
            if current_time - t < self.rate_limit_window
        ]

        # Verificar límite por IP
        if len(self.connection_attempts[client_ip]) >= self.max_sessions_per_ip:
            return False, f"Límite de {self.max_sessions_per_ip} sesiones por IP"

        # Registrar intento
        self.connection_attempts[client_ip].append(current_time)
        self.active_sessions[session_id] = client_ip

        return True, "OK"

    def disconnect_session(self, session_id: str):
        """Limpia una sesión desconectada"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

    def get_session_count_for_ip(self, client_ip: str) -> int:
        """Obtiene número de sesiones activas para una IP"""
        return sum(1 for ip in self.active_sessions.values() if ip == client_ip)


class StreamingMetrics:
    """Métricas de rendimiento para streaming"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.frames_sent = 0
        self.messages_sent = 0
        self.bytes_sent = 0
        self.processing_times = []
        self.websocket_errors = 0
        self.start_time = time.time()

    def add_frame_sent(self, frame_size_bytes: int = 0):
        self.frames_sent += 1
        self.bytes_sent += frame_size_bytes

    def add_message_sent(self, message_size_bytes: int = 0):
        self.messages_sent += 1
        self.bytes_sent += message_size_bytes

    def add_processing_time(self, processing_time: float):
        self.processing_times.append(processing_time)

    def add_websocket_error(self):
        self.websocket_errors += 1

    def get_stats(self) -> Dict[str, Any]:
        current_time = time.time()
        total_time = current_time - self.start_time

        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )

        return {
            "total_time": total_time,
            "frames_sent": self.frames_sent,
            "messages_sent": self.messages_sent,
            "bytes_sent": self.bytes_sent,
            "websocket_errors": self.websocket_errors,
            "fps": self.frames_sent / max(total_time, 1),
            "messages_per_second": self.messages_sent / max(total_time, 1),
            "avg_processing_time": avg_processing_time,
            "bandwidth_mbps": (self.bytes_sent * 8) / (max(total_time, 1) * 1024 * 1024)
        }


class ConnectionManager:
    """Gestor principal de conexiones WebSocket"""

    def __init__(self):
        self.sessions: Dict[str, StreamingSession] = {}
        self.security = SecurityManager()
        self.metrics = StreamingMetrics()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="websocket")
        self._cleanup_task = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Inicia tarea de limpieza periódica"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # Cada minuto
                    await self._cleanup_inactive_sessions()
                except Exception as e:
                    logger.error(f"❌ Error en cleanup task: {str(e)}")

        # MÉTODO CORREGIDO:
        def start_cleanup():
            try:
                loop = asyncio.get_running_loop()
                self._loop = loop
                self._cleanup_task = loop.create_task(cleanup_loop())
                logger.info("✅ Cleanup task iniciado")
            except RuntimeError:
                # No hay loop activo, se iniciará después
                logger.info("ℹ️ Loop no disponible, cleanup se iniciará después")
                pass

        start_cleanup()

    async def start_cleanup_if_needed(self):
        """Inicia cleanup si no está iniciado"""
        if self._cleanup_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._loop = loop

                async def cleanup_loop():
                    while True:
                        try:
                            await asyncio.sleep(60)  # Cada minuto
                            await self._cleanup_inactive_sessions()
                        except Exception as e:
                            logger.error(f"❌ Error en cleanup task: {str(e)}")

                self._cleanup_task = loop.create_task(cleanup_loop())
                logger.info("✅ Cleanup task iniciado desde WebSocket")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo iniciar cleanup: {str(e)}")

    async def connect(self, websocket: WebSocket, session_id: str, client_ip: str = "unknown") -> bool:
        """
        Conecta un nuevo cliente WebSocket

        Returns:
            True si la conexión fue exitosa
        """
        try:
            # Validar seguridad
            can_connect, reason = self.security.can_connect(client_ip, session_id)
            if not can_connect:
                logger.warning(f"🚫 Conexión rechazada para {session_id} ({client_ip}): {reason}")
                await websocket.close(code=4001, reason=reason)
                return False

            # Aceptar conexión WebSocket
            await websocket.accept()

            # Crear sesión
            session = StreamingSession(
                session_id=session_id,
                websocket=websocket,
                status=StreamingStatus.CONNECTED,
                last_activity=time.time()
            )

            self.sessions[session_id] = session

            logger.info(f"🔌 Cliente conectado: {session_id} desde {client_ip}")

            # Enviar confirmación de conexión
            await self.send_message(session_id, {
                "type": "connection_established",
                "session_id": session_id,
                "status": StreamingStatus.CONNECTED.value,
                "server_time": time.time()
            })

            return True

        except Exception as e:
            logger.error(f"❌ Error conectando {session_id}: {str(e)}")
            self.security.disconnect_session(session_id)
            return False

    async def disconnect(self, session_id: str):
        """Desconecta un cliente"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.status = StreamingStatus.DISCONNECTED
            session.should_stop = True

            # Cerrar WebSocket si está abierto
            if session.websocket and not session.websocket.client_state.DISCONNECTED:
                try:
                    await session.websocket.close()
                except:
                    pass

            # Limpiar sesión
            del self.sessions[session_id]
            self.security.disconnect_session(session_id)

            logger.info(f"🔌 Cliente desconectado: {session_id}")

    async def send_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """
        Envía mensaje a un cliente específico

        Returns:
            True si se envió exitosamente
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        if not session.websocket:
            return False

        try:
            message_str = json.dumps(message)
            await session.websocket.send_text(message_str)

            # Actualizar métricas
            session.last_activity = time.time()
            self.metrics.add_message_sent(len(message_str))

            return True

        except WebSocketDisconnect:
            logger.info(f"🔌 Cliente {session_id} desconectado durante envío")
            await self.disconnect(session_id)
            return False
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje a {session_id}: {str(e)}")
            self.metrics.add_websocket_error()
            return False

    async def broadcast_to_session(self, session_id: str, message_type: str, data: Dict[str, Any]) -> bool:
        """Envía datos de procesamiento al cliente"""
        return await self.send_message(session_id, {
            "type": message_type,
            "data": data,
            "timestamp": time.time()
        })

    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Obtiene una sesión por ID"""
        return self.sessions.get(session_id)

    def update_session_status(self, session_id: str, status: StreamingStatus):
        """Actualiza el estado de una sesión"""
        if session_id in self.sessions:
            self.sessions[session_id].status = status
            logger.debug(f"📊 Sesión {session_id}: {status.value}")

    def get_active_sessions_info(self) -> Dict[str, Any]:
        """Obtiene información de todas las sesiones activas"""
        sessions_info = {}
        current_time = time.time()

        for session_id, session in self.sessions.items():
            sessions_info[session_id] = {
                "session_id": session_id,
                "status": session.status.value,
                "connected": session.websocket is not None,
                "total_frames": session.total_frames,
                "processed_frames": session.processed_frames,
                "progress_percent": (
                    (session.processed_frames / max(session.total_frames, 1)) * 100
                    if session.total_frames > 0 else 0
                ),
                "detections_count": len(session.detections),
                "unique_plates_count": len(session.unique_plates),
                "processing_speed": session.processing_speed,
                "uptime": (
                    current_time - session.start_time
                    if session.start_time else 0
                ),
                "last_activity": (
                    current_time - session.last_activity
                    if session.last_activity else 0
                ),
                "is_paused": session.is_paused,
                "video_info": session.video_info
            }

        return {
            "total_sessions": len(self.sessions),
            "active_connections": len([s for s in self.sessions.values() if s.websocket]),
            "sessions": sessions_info,
            "system_metrics": self.metrics.get_stats()
        }

    async def _cleanup_inactive_sessions(self):
        """Limpia sesiones inactivas"""
        current_time = time.time()
        inactive_sessions = []

        for session_id, session in self.sessions.items():
            # Sesión inactiva por más de 30 minutos
            if (session.last_activity and
                    current_time - session.last_activity > 1800):
                inactive_sessions.append(session_id)

            # Sesión en error por más de 5 minutos
            elif (session.status == StreamingStatus.ERROR and
                  session.last_activity and
                  current_time - session.last_activity > 300):
                inactive_sessions.append(session_id)

        # Limpiar sesiones inactivas
        for session_id in inactive_sessions:
            logger.info(f"🧹 Limpiando sesión inactiva: {session_id}")
            await self.disconnect(session_id)

    async def pause_session(self, session_id: str) -> bool:
        """Pausa el procesamiento de una sesión"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.is_paused = True
            session.status = StreamingStatus.PAUSED

            await self.send_message(session_id, {
                "type": "processing_paused",
                "timestamp": time.time()
            })

            logger.info(f"⏸️ Sesión pausada: {session_id}")
            return True
        return False

    async def resume_session(self, session_id: str) -> bool:
        """Reanuda el procesamiento de una sesión"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.is_paused = False
            session.status = StreamingStatus.PROCESSING

            await self.send_message(session_id, {
                "type": "processing_resumed",
                "timestamp": time.time()
            })

            logger.info(f"▶️ Sesión reanudada: {session_id}")
            return True
        return False

    async def stop_session(self, session_id: str) -> bool:
        """Detiene el procesamiento de una sesión"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.should_stop = True
            session.status = StreamingStatus.CANCELLED

            await self.send_message(session_id, {
                "type": "processing_stopped",
                "timestamp": time.time()
            })

            logger.info(f"⏹️ Sesión detenida: {session_id}")
            return True
        return False

    def cleanup(self):
        """Limpia recursos del manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self.executor:
            self.executor.shutdown(wait=False)


# Instancia global del gestor
connection_manager = ConnectionManager()