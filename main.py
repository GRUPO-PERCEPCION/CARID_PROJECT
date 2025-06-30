"""
🚗 CARID ALPR - Sistema Limpio con Streaming WebSocket
Versión completamente reescrita para máxima eficiencia
"""

from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import json
import time
from loguru import logger
import sys
import os

# Configuración de logging
from config.settings import settings

# Configurar loguru
logger.remove()
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
)

# Logging a archivo
if settings.log_file:
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
    logger.add(settings.log_file, level=settings.log_level, rotation="10 MB", retention="7 days")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""

    # 🚀 STARTUP
    logger.info("🚀 Iniciando CARID ALPR con Streaming")

    try:
        # Crear directorios
        settings.create_directories()

        # Validar modelos
        validation = settings.validate_model_files()
        if not validation["plate_model_exists"] or not validation["char_model_exists"]:
            logger.error("❌ Archivos de modelos no encontrados")
            raise HTTPException(status_code=500, detail="Modelos no encontrados")

        # Cargar modelos
        from models.model_manager import model_manager
        if not model_manager.load_models():
            logger.error("❌ Error cargando modelos")
            raise HTTPException(status_code=500, detail="Error cargando modelos")

        # Warmup
        model_manager.warmup_models()

        logger.success("✅ Sistema iniciado correctamente")
        logger.info(f"🌐 API: http://{settings.host}:{settings.port}")
        logger.info(f"📚 Docs: http://{settings.host}:{settings.port}/docs")
        logger.info(f"🔌 WebSocket: ws://{settings.host}:{settings.port}/api/v1/streaming/ws/{{session_id}}")

    except Exception as e:
        logger.error(f"❌ Error en startup: {str(e)}")
        raise

    yield

    # 🛑 SHUTDOWN
    logger.info("🛑 Cerrando CARID ALPR")

    try:
        # Limpiar archivos temporales
        from services.file_service import file_service
        file_service.cleanup_old_files(0)
        logger.info("✅ Limpieza completada")
    except Exception as e:
        logger.warning(f"⚠️ Error en limpieza: {str(e)}")


# 🎯 CREAR APLICACIÓN
app = FastAPI(
    title="CARID - Sistema ALPR con Streaming",
    description="""
    ## 🚗 CARID - Reconocimiento de Placas con Streaming en Tiempo Real
    
    ### 🎯 Características:
    - **🔍 Detección de placas** con YOLOv8
    - **📖 Reconocimiento de caracteres** de alta precisión  
    - **🎬 Streaming en tiempo real** con WebSocket
    - **✅ Validación de formato** para placas peruanas
    - **⚡ Procesamiento GPU** con CUDA
    
    ### 🔌 Endpoints WebSocket:
    - `ws://localhost:8000/api/v1/streaming/ws/{session_id}` - Streaming principal
    - `ws://localhost:8000/api/v1/streaming/test/{session_id}` - Pruebas
    - `ws://localhost:8000/simple-test` - Test ultra-simple
    
    ### 📡 Endpoints REST:
    - `POST /api/v1/streaming/upload` - Subir video para streaming
    - `GET /api/v1/streaming/sessions` - Listar sesiones activas
    - `GET /api/v1/streaming/health` - Health check
    - `POST /api/v1/detect/image` - Detección en imágenes
    
    ### 🚀 Uso Rápido:
    1. Conectar WebSocket: `ws://localhost:8000/api/v1/streaming/ws/mi_sesion`
    2. Subir video: `POST /api/v1/streaming/upload`
    3. Enviar: `{"type": "start_processing"}`
    4. Recibir frames procesados en tiempo real
    """,
    version=settings.app_version,
    lifespan=lifespan
)

# 🌐 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes temporalmente
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


@app.middleware("http")
async def enhanced_ngrok_middleware(request: Request, call_next):
    """Middleware mejorado para manejar ngrok y CORS"""

    # Log de request para debugging
    logger.info(f"🌐 Request: {request.method} {request.url}")
    logger.info(f"📡 Headers: {dict(request.headers)}")

    # Procesar request
    response = await call_next(request)

    # Headers específicos para ngrok
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, ngrok-skip-browser-warning"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["ngrok-skip-browser-warning"] = "true"

    # Headers adicionales para evitar 403
    response.headers["Access-Control-Max-Age"] = "86400"
    response.headers["Vary"] = "Origin"

    logger.info(f"✅ Response: {response.status_code}")

    return response

@app.options("/{full_path:path}")
async def options_handler(request: Request):
    """Manejar requests OPTIONS para CORS preflight"""
    return {
        "message": "OK",
        "method": "OPTIONS",
        "path": request.url.path
    }

# 📁 Archivos estáticos
try:
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
except Exception as e:
    logger.warning(f"⚠️ Error montando archivos estáticos: {str(e)}")

# 🛣️ INCLUIR ROUTERS
logger.info("🛣️ Registrando rutas...")

# Routers básicos
from api.routes import health, detection
from api.routes.video import video_router

app.include_router(health.router)
app.include_router(detection.router)
app.include_router(video_router)

# 🎬 ROUTER DE STREAMING (NUEVO)
from api.routes.streaming import streaming_router
app.include_router(streaming_router)

logger.info("✅ Rutas registradas:")
logger.info("   📊 Health: /api/v1/health/*")
logger.info("   🔍 Detection: /api/v1/detect/*")
logger.info("   🎬 Video: /api/v1/video/*")
logger.info("   🌐 Streaming: /api/v1/streaming/* (NUEVO)")


# 🏠 ENDPOINT RAÍZ
@app.get("/", tags=["Root"])
async def root():
    """Información principal de la API"""

    try:
        from models.model_manager import model_manager
        models_loaded = model_manager.is_loaded
    except:
        models_loaded = False

    return {
        "service": "🚗 CARID - Sistema ALPR con Streaming",
        "version": settings.app_version,
        "status": "running",
        "models_loaded": models_loaded,
        "streaming_enabled": settings.streaming_enabled,
        "endpoints": {
            "documentation": "/docs",
            "health": "/api/v1/health",
            "detect_image": "/api/v1/detect/image",
            "process_video": "/api/v1/video/detect",
            "streaming_websocket": "/api/v1/streaming/ws/{session_id}",
            "streaming_upload": "/api/v1/streaming/upload",
            "streaming_health": "/api/v1/streaming/health"
        },
        "websocket_examples": {
            "main_streaming": "ws://localhost:8000/api/v1/streaming/ws/my_session",
            "test_connection": "ws://localhost:8000/api/v1/streaming/test/test123",
            "simple_test": "ws://localhost:8000/simple-test"
        },
        "quick_start": {
            "step_1": "Conecta WebSocket a ws://localhost:8000/api/v1/streaming/ws/mi_sesion",
            "step_2": "Sube video: POST /api/v1/streaming/upload con session_id=mi_sesion",
            "step_3": "Envía: {\"type\": \"start_processing\"}",
            "step_4": "Recibe frames procesados en tiempo real"
        },
        "timestamp": time.time()
    }

@app.get("/api/v1/test-ngrok", tags=["Testing"])
async def test_ngrok():
    """Endpoint simple para probar ngrok"""
    return {
        "success": True,
        "message": "✅ Ngrok funcionando correctamente",
        "timestamp": time.time(),
        "service": "CARID ALPR"
    }

# 🧪 WEBSOCKET DE PRUEBA ULTRA-SIMPLE
@app.websocket("/simple-test")
async def simple_test_websocket(websocket: WebSocket):
    """WebSocket de prueba ultra-simple para verificar conectividad básica"""

    try:
        logger.info("🧪 Intentando conectar WebSocket simple...")
        await websocket.accept()
        logger.success("✅ WebSocket simple conectado")

        # Mensaje de bienvenida
        welcome = {
            "type": "connected",
            "message": "🎉 WebSocket funcionando perfectamente",
            "service": "CARID ALPR",
            "timestamp": time.time(),
            "instructions": {
                "send_ping": "Envía: {\"type\": \"ping\"}",
                "send_message": "Envía cualquier texto para recibir eco"
            }
        }
        await websocket.send_text(json.dumps(welcome))

        # Loop principal
        message_count = 0
        while True:
            try:
                # Recibir mensaje
                data = await websocket.receive_text()
                message_count += 1

                logger.info(f"📥 Mensaje {message_count}: {data[:50]}...")

                # Intentar parsear como JSON
                try:
                    message = json.loads(data)
                    message_type = message.get("type", "unknown")

                    if message_type == "ping":
                        response = {
                            "type": "pong",
                            "message": "🏓 Pong! WebSocket funcionando",
                            "timestamp": time.time(),
                            "message_count": message_count
                        }
                    else:
                        response = {
                            "type": "json_echo",
                            "received": message,
                            "message": f"📨 Mensaje JSON #{message_count} recibido",
                            "timestamp": time.time()
                        }

                except json.JSONDecodeError:
                    # No es JSON, responder como texto
                    response = {
                        "type": "text_echo",
                        "original_text": data,
                        "message": f"📝 Mensaje de texto #{message_count} recibido",
                        "echo": f"Echo: {data}",
                        "timestamp": time.time()
                    }

                await websocket.send_text(json.dumps(response))

            except Exception as e:
                logger.error(f"❌ Error en loop simple: {str(e)}")
                break

    except Exception as e:
        logger.error(f"❌ Error en WebSocket simple: {str(e)}")
    finally:
        logger.info("🧪 WebSocket simple desconectado")


# 📊 ENDPOINT DE ESTADO RÁPIDO
@app.get("/status", tags=["System"])
async def quick_status():
    """Estado rápido del sistema"""

    try:
        # Importar streaming router para obtener sesiones
        from api.routes.streaming import active_sessions

        return {
            "status": "running",
            "timestamp": time.time(),
            "streaming": {
                "active_sessions": len(active_sessions),
                "max_sessions": settings.max_websocket_connections,
                "websocket_url": f"ws://{settings.host}:{settings.port}/api/v1/streaming/ws/{{session_id}}"
            },
            "models": {
                "streaming_enabled": settings.streaming_enabled,
                "max_file_size_mb": settings.max_file_size,
                "cuda_available": settings.is_cuda_available
            },
            "test_endpoints": {
                "simple_websocket": f"ws://{settings.host}:{settings.port}/simple-test",
                "streaming_test": f"ws://{settings.host}:{settings.port}/api/v1/streaming/test/test123",
                "health_check": "/api/v1/streaming/health"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


# 🔧 MANEJADORES DE ERRORES
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de errores"""
    logger.error(f"❌ Error global: {str(exc)}")

    return {
        "success": False,
        "error": {
            "type": "InternalServerError",
            "message": str(exc) if settings.debug else "Error interno del servidor"
        },
        "endpoint": str(request.url.path),
        "timestamp": time.time()
    }


if __name__ == "__main__":
    # 🎬 MENSAJE DE INICIO
    logger.info("🚗 CARID ALPR - Sistema de Streaming Reescrito")
    logger.info("=" * 60)
    logger.info("🎯 Características disponibles:")
    logger.info("   ✅ Detección de placas en imágenes")
    logger.info("   ✅ Procesamiento de videos")
    logger.info("   ✅ 🆕 Streaming WebSocket en tiempo real")
    logger.info("   ✅ 🆕 Sistema de sesiones simplificado")
    logger.info("   ✅ 🆕 Endpoints de prueba incluidos")
    logger.info("   ✅ API REST completa")
    logger.info("=" * 60)
    logger.info("🔌 Endpoints WebSocket disponibles:")
    logger.info("   🎬 Principal: ws://localhost:8000/api/v1/streaming/ws/{session_id}")
    logger.info("   🧪 Prueba: ws://localhost:8000/api/v1/streaming/test/{session_id}")
    logger.info("   🏃 Simple: ws://localhost:8000/simple-test")
    logger.info("=" * 60)
    logger.info("📡 Endpoints REST clave:")
    logger.info("   📚 Documentación: http://localhost:8000/docs")
    logger.info("   🏥 Health Check: http://localhost:8000/api/v1/streaming/health")
    logger.info("   📊 Estado Rápido: http://localhost:8000/status")
    logger.info("   📤 Upload Video: POST http://localhost:8000/api/v1/streaming/upload")
    logger.info("=" * 60)

    # 🚀 EJECUTAR SERVIDOR
    logger.info("🚀 Iniciando servidor...")

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
        access_log=True,
        ws="auto",  # ✅ CRÍTICO: Habilitar WebSocket
        timeout_keep_alive=60,
        limit_max_requests=2000
    )