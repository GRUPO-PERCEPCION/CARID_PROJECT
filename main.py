from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
import sys
import os

from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Configurar logging
from config.settings import settings

# Configurar loguru
logger.remove()  # Remover handler por defecto
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Agregar logging a archivo si está configurado
if settings.log_file:
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

# Importar después de configurar logging
from models.model_manager import model_manager
from api.routes import health, detection
from api.routes.video import video_router
from api.routes.streaming import streaming_router


# 🔧 MIDDLEWARE PERSONALIZADO PARA ARCHIVOS GRANDES
class LargeFileMiddleware(BaseHTTPMiddleware):
    """Middleware para manejar archivos grandes"""

    async def dispatch(self, request: Request, call_next):
        # Verificar tamaño del contenido
        content_length = request.headers.get("content-length")

        if content_length:
            content_length = int(content_length)
            max_size = settings.max_file_size * 1024 * 1024  # Convertir MB a bytes

            if content_length > max_size:
                return JSONResponse(
                    status_code=413,
                    content={
                        "success": False,
                        "error": {
                            "type": "PayloadTooLarge",
                            "message": f"Archivo muy grande. Tamaño máximo: {settings.max_file_size}MB, recibido: {content_length / (1024 * 1024):.1f}MB",
                            "max_size_mb": settings.max_file_size,
                            "received_size_mb": round(content_length / (1024 * 1024), 1)
                        }
                    }
                )

        response = await call_next(request)
        return response


# 🌐 MIDDLEWARE PARA WEBSOCKET Y STREAMING
class StreamingMiddleware(BaseHTTPMiddleware):
    """Middleware específico para optimizar streaming"""

    async def dispatch(self, request: Request, call_next):
        # Optimizaciones para rutas de streaming
        if request.url.path.startswith("/api/v1/streaming"):
            # Headers específicos para streaming
            request.headers.__dict__.setdefault("_list", [])

            # Configurar timeout extendido para streaming
            if "upload" in request.url.path or "start-session" in request.url.path:
                # Requests de upload pueden tardar más
                pass

        response = await call_next(request)

        # Headers específicos para streaming
        if request.url.path.startswith("/api/v1/streaming"):
            response.headers["X-Streaming-Service"] = "CARID-ALPR"
            response.headers["X-Service-Version"] = settings.app_version

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación con streaming"""

    # Startup
    logger.info("🚀 Iniciando CARID ALPR API con Streaming...")

    try:
        # Crear directorios necesarios (incluyendo streaming)
        logger.info("📁 Creando directorios...")
        settings.create_directories()

        # Validar archivos de modelos
        logger.info("🔍 Validando archivos de modelos...")
        validation = settings.validate_model_files()

        if not validation["models_dir_exists"]:
            logger.error("❌ Directorio de modelos no existe: ./models_trained")
            raise HTTPException(
                status_code=500,
                detail="Directorio de modelos no encontrado"
            )

        if not validation["plate_model_exists"]:
            logger.error(f"❌ Modelo de placas no encontrado: {settings.plate_model_path}")
            raise HTTPException(
                status_code=500,
                detail="Modelo de detección de placas no encontrado"
            )

        if not validation["char_model_exists"]:
            logger.error(f"❌ Modelo de caracteres no encontrado: {settings.char_model_path}")
            raise HTTPException(
                status_code=500,
                detail="Modelo de reconocimiento de caracteres no encontrado"
            )

        # Cargar modelos
        logger.info("🤖 Cargando modelos YOLOv8...")
        if not model_manager.load_models():
            logger.error("❌ Error cargando modelos")
            raise HTTPException(
                status_code=500,
                detail="Error al cargar los modelos YOLOv8"
            )

        # Realizar warmup opcional
        logger.info("🔥 Realizando warmup de modelos...")
        model_manager.warmup_models()

        # ✅ VALIDAR CONFIGURACIÓN DE STREAMING
        logger.info("🌐 Validando configuración de streaming...")
        streaming_validation = settings.validate_streaming_settings()

        if streaming_validation["all_valid"]:
            logger.success("✅ Configuración de streaming válida")
        else:
            logger.warning("⚠️ Algunas configuraciones de streaming pueden no ser óptimas")
            for key, valid in streaming_validation.items():
                if not valid and key != "all_valid":
                    logger.warning(f"   ⚠️ {key}: configuración subóptima")

        # ✅ INICIALIZAR SERVICIOS DE STREAMING
        logger.info("🎬 Inicializando servicios de streaming...")

        # Importar e inicializar servicios de streaming
        from services.streaming_service import streaming_service
        from api.websocket_manager import connection_manager

        logger.success("✅ Servicios de streaming inicializados")

        logger.success("✅ CARID ALPR API con Streaming iniciada exitosamente")
        logger.info(f"🌐 API disponible en: http://{settings.host}:{settings.port}")
        logger.info(f"📚 Documentación disponible en: http://{settings.host}:{settings.port}/docs")
        logger.info(f"📂 Tamaño máximo de archivos: {settings.max_file_size}MB")
        logger.info(f"🎬 Streaming habilitado: {'✅' if settings.streaming_enabled else '❌'}")
        logger.info(f"🔌 Máx. conexiones WebSocket: {settings.max_websocket_connections}")

    except Exception as e:
        logger.error(f"❌ Error durante startup: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Cerrando CARID ALPR API...")

    try:
        # ✅ LIMPIAR SERVICIOS DE STREAMING
        logger.info("🧹 Limpiando servicios de streaming...")

        # Limpiar connection manager
        from api.websocket_manager import connection_manager
        connection_manager.cleanup()

        # Limpiar streaming service
        from services.streaming_service import streaming_service
        streaming_service.cleanup()

        logger.info("✅ Servicios de streaming limpiados")

        # Limpiar archivos temporales en shutdown
        from services.file_service import file_service
        file_service.cleanup_old_files(0)
        logger.info("🗑️ Archivos temporales limpiados")

    except Exception as e:
        logger.warning(f"⚠️ Error durante limpieza: {str(e)}")

    logger.info("✅ API cerrada exitosamente")


# Crear aplicación FastAPI
app = FastAPI(
    title="CARID - Sistema ALPR con Streaming",
    description="""
    ## 🚗 CARID - Sistema de Reconocimiento Automático de Placas Vehiculares

    Sistema avanzado de detección y reconocimiento de placas vehiculares usando YOLOv8 con **streaming en tiempo real**.

    ### 🎯 Características Principales:
    - **Detección precisa** de placas vehiculares con YOLOv8
    - **Reconocimiento de caracteres** con alta precisión
    - **🆕 Streaming en tiempo real** con WebSocket
    - **🆕 Monitoreo en vivo** del procesamiento de video
    - **🆕 Control interactivo** (pause/resume/stop)
    - **Procesamiento de videos** con tracking inteligente avanzado
    - **Validación de formato** para placas peruanas
    - **Procesamiento optimizado** con GPU CUDA
    - **API REST completa** y documentada
    - **✅ Soporte para archivos hasta 150MB**

    ### 🔧 Tecnologías:
    - **YOLOv8** para detección y reconocimiento
    - **PyTorch** con soporte CUDA 11.8
    - **FastAPI + WebSocket** para streaming en tiempo real
    - **OpenCV** para procesamiento de imágenes y videos

    ### 📋 Endpoints Principales:

    **🔍 Detección Estática:**
    - `POST /api/v1/detect/image` - Detección completa en imágenes
    - `POST /api/v1/detect/image/quick` - Detección rápida en imágenes
    - `POST /api/v1/video/detect` - Detección en videos (sin streaming)

    **🎬 Streaming en Tiempo Real:**
    - `WS /api/v1/streaming/ws/{session_id}` - **NUEVO:** WebSocket para streaming
    - `POST /api/v1/streaming/start-session` - **NUEVO:** Iniciar sesión de streaming
    - `GET /api/v1/streaming/sessions` - **NUEVO:** Listar sesiones activas
    - `POST /api/v1/streaming/sessions/{session_id}/control` - **NUEVO:** Control de sesión

    **🏥 Monitoreo:**
    - `GET /api/v1/health` - Health checks básicos
    - `GET /api/v1/streaming/health` - **NUEVO:** Health check de streaming
    - `GET /api/v1/streaming/stats` - **NUEVO:** Estadísticas de streaming
    - `GET /docs` - Esta documentación

    ### 🚀 Etapa Actual: 4 - Streaming en Tiempo Real
    ✅ Detección y reconocimiento en imágenes  
    ✅ Validación de formatos peruanos  
    ✅ Procesamiento de videos frame por frame  
    ✅ Tracking inteligente con doble confianza  
    ✅ Sistema anti-duplicación avanzado  
    ✅ **NUEVO:** Streaming en tiempo real con WebSocket  
    ✅ **NUEVO:** Monitoreo en vivo del procesamiento  
    ✅ **NUEVO:** Control interactivo de sesiones  
    ✅ **NUEVO:** Calidad adaptativa automática  
    ✅ **NUEVO:** Timeline de detecciones en tiempo real

    ### 🌐 Funcionalidades de Streaming:
    - **WebSocket Full-Duplex**: Comunicación bidireccional en tiempo real
    - **Calidad Adaptativa**: Ajusta automáticamente según ancho de banda
    - **Control Interactivo**: Pause/Resume/Stop durante procesamiento
    - **Timeline en Tiempo Real**: Ve las detecciones conforme aparecen
    - **Múltiples Sesiones**: Hasta 20 sesiones simultáneas
    - **Frames Anotados**: Ve el video con bounding boxes en tiempo real
    - **Estadísticas en Vivo**: Progreso, velocidad, detecciones actualizadas
    - **Thumbnails**: Previsualización rápida de frames
    - **Exportación**: Descarga resultados en JSON/CSV

    ### 📊 Experiencia de Usuario:
    **Lo que ve el usuario en tiempo real:**
    - 🎬 **Video reproduciéndose** con bounding boxes dinámicos
    - 📋 **Lista actualizada** de placas detectadas con timestamp
    - 📊 **Progreso en tiempo real** con velocidad de procesamiento
    - 🎯 **Información detallada** de confianza y validez
    - 🎮 **Controles interactivos** para pausar/reanudar
    - 📈 **Métricas en vivo** de calidad y rendimiento
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 🔧 CONFIGURAR MIDDLEWARES PARA ARCHIVOS GRANDES Y STREAMING

# 1. Middleware para streaming (primero para optimizar)
# app.add_middleware(StreamingMiddleware)  # Temporalmente deshabilitado

# 2. Middleware personalizado para archivos grandes
# app.add_middleware(LargeFileMiddleware)  # Temporalmente deshabilitado

# 3. CORS con configuración extendida para WebSocket
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight por 1 hora
)

# Montar archivos estáticos
try:
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    logger.info(f"📁 Archivos estáticos montados en: /static -> {settings.static_dir}")
except Exception as e:
    logger.warning(f"⚠️ No se pudieron montar archivos estáticos: {str(e)}")

# ✅ INCLUIR RUTAS (ORDEN IMPORTANTE)
app.include_router(health.router)
app.include_router(detection.router)
app.include_router(video_router)
app.include_router(streaming_router)  # ✅ NUEVO: Rutas de streaming

logger.info("🛣️ Rutas registradas:")
logger.info("   📊 Health: /api/v1/health/*")
logger.info("   🔍 Detection: /api/v1/detect/*")
logger.info("   🎬 Video: /api/v1/video/*")
logger.info("   🌐 Streaming: /api/v1/streaming/* (NUEVO)")


# Endpoint raíz actualizado con información de streaming
@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz de la API con información completa incluyendo streaming"""

    # Obtener estado de los modelos
    try:
        models_status = model_manager.get_model_info() if model_manager.is_loaded else {"models_loaded": False}
    except:
        models_status = {"models_loaded": False}

    # Obtener estado del streaming
    try:
        from api.websocket_manager import connection_manager
        streaming_info = connection_manager.get_active_sessions_info()
    except:
        streaming_info = {"total_sessions": 0, "active_connections": 0}

    return {
        "message": "🚗 CARID - Sistema ALPR con Streaming en Tiempo Real",
        "version": settings.app_version,
        "status": "running",
        "etapa_actual": "4 - Streaming en Tiempo Real Completo",

        "funcionalidades": {
            "deteccion_placas_imagenes": "✅ Disponible",
            "reconocimiento_caracteres": "✅ Disponible",
            "validacion_formato": "✅ Disponible",
            "procesamiento_videos": "✅ Disponible",
            "tracking_avanzado": "✅ Doble confianza + estabilidad",
            "anti_duplicacion": "✅ Sistema inteligente",
            "archivos_grandes": "✅ Hasta 150MB",
            "streaming_tiempo_real": "✅ **NUEVO** WebSocket + Monitoreo en vivo",
            "control_interactivo": "✅ **NUEVO** Pause/Resume/Stop en tiempo real",
            "calidad_adaptativa": "✅ **NUEVO** Optimización automática",
            "sesiones_multiples": "✅ **NUEVO** Hasta 20 sesiones simultáneas"
        },

        "modelos": {
            "cargados": models_status.get("models_loaded", False),
            "dispositivo": models_status.get("device", "unknown"),
            "detector_placas": models_status.get("plate_detector_loaded", False),
            "reconocedor_caracteres": models_status.get("char_recognizer_loaded", False)
        },

        "streaming": {
            "habilitado": settings.streaming_enabled,
            "sesiones_activas": streaming_info.get("total_sessions", 0),
            "conexiones_activas": streaming_info.get("active_connections", 0),
            "max_sesiones": settings.max_websocket_connections,
            "calidad_adaptativa": settings.streaming_adaptive_quality,
            "websocket_endpoint": "/api/v1/streaming/ws/{session_id}"
        },

        "endpoints": {
            # Endpoints existentes
            "deteccion_imagen": "/api/v1/detect/image",
            "deteccion_rapida": "/api/v1/detect/image/quick",
            "deteccion_video": "/api/v1/video/detect",
            "deteccion_video_rapida": "/api/v1/video/detect/quick",

            # NUEVOS endpoints de streaming
            "websocket_streaming": "/api/v1/streaming/ws/{session_id}",
            "iniciar_streaming": "/api/v1/streaming/start-session",
            "control_streaming": "/api/v1/streaming/sessions/{session_id}/control",
            "sesiones_streaming": "/api/v1/streaming/sessions",
            "estadisticas_streaming": "/api/v1/streaming/stats",
            "health_streaming": "/api/v1/streaming/health",

            # Health y docs
            "health_check": "/api/v1/health",
            "documentacion": "/docs",
            "estadisticas": "/api/v1/detect/stats"
        },

        "configuracion": {
            "max_file_size_mb": settings.max_file_size,
            "formatos_imagenes": ["jpg", "jpeg", "png"],
            "formatos_videos": ["mp4", "avi", "mov", "mkv", "webm"],
            "max_video_duration": settings.max_video_duration,
            "cuda_disponible": models_status.get("cuda_available", False),
            "streaming_frame_quality": settings.streaming_frame_quality,
            "streaming_max_size": settings.streaming_frame_max_size
        },

        "novedades_streaming": {
            "websocket_bidireccional": "Comunicación en tiempo real full-duplex",
            "monitoreo_en_vivo": "Ve el procesamiento frame por frame",
            "control_interactivo": "Pausa/reanuda/detiene durante procesamiento",
            "calidad_adaptativa": "Ajuste automático según ancho de banda",
            "timeline_detecciones": "Timeline en tiempo real de placas detectadas",
            "sesiones_multiples": "Hasta 20 usuarios simultáneos",
            "frames_anotados": "Video con bounding boxes en tiempo real",
            "thumbnails": "Previsualización rápida optimizada",
            "exportacion_avanzada": "Descarga resultados en JSON/CSV",
            "metricas_tiempo_real": "Estadísticas de rendimiento en vivo"
        },

        "arquitectura_streaming": {
            "frontend_conecta": "WebSocket a /api/v1/streaming/ws/{session_id}",
            "backend_procesa": "Frame por frame con YOLOv8",
            "envio_tiempo_real": "Frames + detecciones vía WebSocket",
            "frontend_muestra": "Video con overlays + lista actualizada",
            "control_bidireccional": "Frontend puede pausar/reanudar backend",
            "calidad_dinamica": "Ajuste automático según velocidad conexión"
        },

        "flujo_usuario": {
            "paso_1": "Frontend conecta WebSocket",
            "paso_2": "Usuario sube video vía REST API",
            "paso_3": "Backend inicia procesamiento frame por frame",
            "paso_4": "Envío continuo: frame + detecciones + progreso",
            "paso_5": "Frontend muestra video en vivo con bounding boxes",
            "paso_6": "Lista de placas se actualiza en tiempo real",
            "paso_7": "Usuario puede pausar/reanudar a voluntad",
            "paso_8": "Descarga resultados al finalizar"
        }
    }


# Endpoint de información del sistema actualizado
@app.get("/system", tags=["System"])
async def system_info():
    """Información detallada del sistema incluyendo streaming"""
    try:
        from api.dependencies import get_system_info
        system_info = get_system_info()

        # Agregar información específica de streaming
        try:
            from api.websocket_manager import connection_manager
            from services.streaming_service import streaming_service

            streaming_system_info = {
                "websocket_manager": {
                    "active_sessions": len(connection_manager.sessions),
                    "max_sessions": settings.max_websocket_connections,
                    "security_enabled": True
                },
                "streaming_service": {
                    "active_processors": len(streaming_service.quality_managers),
                    "detection_trackers": len(streaming_service.detection_trackers),
                    "thread_pool_workers": streaming_service.executor._max_workers
                },
                "streaming_config": settings.get_streaming_config()
            }

            system_info["streaming"] = streaming_system_info

        except Exception as e:
            system_info["streaming_error"] = str(e)

        return system_info
    except Exception as e:
        logger.error(f"❌ Error obteniendo info del sistema: {str(e)}")
        return {"error": "No se pudo obtener información del sistema"}


# Endpoint específico para estado de streaming
@app.get("/streaming-status", tags=["Streaming"])
async def streaming_status():
    """Estado rápido del sistema de streaming"""
    try:
        from api.websocket_manager import connection_manager
        from services.streaming_service import streaming_service

        sessions_info = connection_manager.get_active_sessions_info()

        return {
            "streaming_enabled": settings.streaming_enabled,
            "active_sessions": sessions_info["total_sessions"],
            "active_connections": sessions_info["active_connections"],
            "max_sessions": settings.max_websocket_connections,
            "capacity_usage_percent": (sessions_info["total_sessions"] / settings.max_websocket_connections) * 100,
            "quality_managers": len(streaming_service.quality_managers),
            "detection_trackers": len(streaming_service.detection_trackers),
            "websocket_endpoint": "/api/v1/streaming/ws/{session_id}",
            "documentation": "/docs#/🎬%20Real-time%20Video%20Streaming",
            "timestamp": __import__('time').time()
        }
    except Exception as e:
        return {
            "streaming_enabled": False,
            "error": str(e),
            "timestamp": __import__('time').time()
        }


# Manejadores de errores mejorados
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"❌ Error global en {request.url}: {str(exc)}")

    # Información adicional en modo debug
    error_detail = str(exc) if settings.debug else "Error interno del servidor"

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "type": "InternalServerError",
                "message": "Error interno del servidor",
                "detail": error_detail
            },
            "endpoint": str(request.url.path),
            "method": request.method,
            "timestamp": str(__import__('datetime').datetime.utcnow().isoformat()),
            "help": "Contacte al administrador si el problema persiste",
            "streaming_status": "check /streaming-status for streaming health"
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    logger.warning(f"⚠️ Error HTTP {exc.status_code} en {request.url}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "type": "HTTPException",
                "message": exc.detail,
                "status_code": exc.status_code
            },
            "endpoint": str(request.url.path),
            "method": request.method,
            "timestamp": str(__import__('datetime').datetime.utcnow().isoformat())
        }
    )


if __name__ == "__main__":
    # Mensaje de inicio actualizado con streaming
    logger.info("🚗 CARID ALPR - Etapa 4: Streaming en Tiempo Real")
    logger.info("=" * 80)
    logger.info("🎯 Funcionalidades disponibles:")
    logger.info("   ✅ Detección de placas en imágenes")
    logger.info("   ✅ Reconocimiento de caracteres")
    logger.info("   ✅ Validación de formatos peruanos")
    logger.info("   ✅ Procesamiento de videos con tracking avanzado")
    logger.info("   ✅ 🆕 Streaming en tiempo real con WebSocket")
    logger.info("   ✅ 🆕 Monitoreo en vivo del procesamiento")
    logger.info("   ✅ 🆕 Control interactivo (pause/resume/stop)")
    logger.info("   ✅ 🆕 Calidad adaptativa automática")
    logger.info("   ✅ 🆕 Timeline de detecciones en tiempo real")
    logger.info("   ✅ 🆕 Sesiones múltiples (hasta 20)")
    logger.info("   ✅ API REST completa")
    logger.info("   ✅ Documentación interactiva")
    logger.info("=" * 80)
    logger.info("🎬 Capacidades de streaming:")
    logger.info("   📹 Procesamiento en tiempo real de MP4, AVI, MOV, MKV, WebM")
    logger.info("   ⏱️ Duración máxima: 10 minutos, hasta 150MB")
    logger.info("   🎯 Doble confianza: Detector + Reconocedor")
    logger.info("   🔄 Tracking estable con validaciones múltiples")
    logger.info("   🌐 WebSocket full-duplex para comunicación bidireccional")
    logger.info("   📊 Monitoreo de progreso, velocidad y detecciones en vivo")
    logger.info("   🎮 Control total: pause/resume/stop durante procesamiento")
    logger.info("   🎨 Frames anotados con bounding boxes en tiempo real")
    logger.info("   📈 Calidad adaptativa según ancho de banda")
    logger.info("   🚀 Procesamiento optimizado y paralelo")
    logger.info("=" * 80)
    logger.info("🌐 Arquitectura de Streaming:")
    logger.info("   1️⃣ Frontend conecta WebSocket: /api/v1/streaming/ws/{session_id}")
    logger.info("   2️⃣ Backend procesa video frame por frame con YOLOv8")
    logger.info("   3️⃣ Envío continuo: frames + detecciones + progreso")
    logger.info("   4️⃣ Frontend muestra video con overlays en tiempo real")
    logger.info("   5️⃣ Lista de placas se actualiza automáticamente")
    logger.info("   6️⃣ Control bidireccional para pausar/reanudar")
    logger.info("=" * 80)

    # Ejecutar servidor con configuración optimizada para streaming
    logger.info("🚀 Iniciando servidor con soporte completo de streaming...")

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
        timeout_keep_alive=60,
        limit_max_requests=2000,
        backlog=4096,
        ws="auto",
    )
