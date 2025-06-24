from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
import sys
import os

from starlette.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""

    # Startup
    logger.info("🚀 Iniciando CARID ALPR API...")

    try:
        # Crear directorios necesarios
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

        logger.success("✅ CARID ALPR API iniciada exitosamente")
        logger.info(f"🌐 API disponible en: http://{settings.host}:{settings.port}")
        logger.info(f"📚 Documentación disponible en: http://{settings.host}:{settings.port}/docs")
        logger.info(f"📂 Tamaño máximo de archivos: {settings.max_file_size}MB")  # ✅ NUEVO LOG

    except Exception as e:
        logger.error(f"❌ Error durante startup: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Cerrando CARID ALPR API...")

    # Limpiar archivos temporales en shutdown
    try:
        from services.file_service import file_service
        file_service.cleanup_old_files(0)
        logger.info("🗑️ Archivos temporales limpiados")
    except Exception as e:
        logger.warning(f"⚠️ Error limpiando archivos temporales: {str(e)}")

    logger.info("✅ API cerrada exitosamente")


# Crear aplicación FastAPI
app = FastAPI(
    title="CARID - Sistema ALPR",
    description="""
    ## 🚗 CARID - Sistema de Reconocimiento Automático de Placas Vehiculares

    Sistema avanzado de detección y reconocimiento de placas vehiculares usando YOLOv8.

    ### 🎯 Características Principales:
    - **Detección precisa** de placas vehiculares con YOLOv8
    - **Reconocimiento de caracteres** con alta precisión
    - **Procesamiento de videos** con tracking inteligente avanzado
    - **Validación de formato** para placas peruanas
    - **Procesamiento optimizado** con GPU CUDA
    - **API REST completa** y documentada
    - **✅ Soporte para archivos hasta 150MB**

    ### 🔧 Tecnologías:
    - **YOLOv8** para detección y reconocimiento
    - **PyTorch** con soporte CUDA 11.8
    - **FastAPI** para la API REST
    - **OpenCV** para procesamiento de imágenes y videos

    ### 📋 Endpoints Principales:
    - `POST /api/v1/detect/image` - Detección completa en imágenes
    - `POST /api/v1/detect/image/quick` - Detección rápida en imágenes
    - `POST /api/v1/video/detect` - **MEJORADO:** Detección en videos con tracking avanzado
    - `POST /api/v1/video/detect/quick` - **MEJORADO:** Detección rápida en videos
    - `GET /api/v1/health` - Health checks
    - `GET /docs` - Esta documentación

    ### 🚀 Etapa Actual: 3+ - Tracking Avanzado de Videos
    ✅ Detección y reconocimiento en imágenes  
    ✅ Validación de formatos peruanos  
    ✅ Procesamiento de videos frame por frame  
    ✅ **NUEVO:** Tracking inteligente con doble confianza  
    ✅ **NUEVO:** Sistema anti-duplicación avanzado  
    ✅ **NUEVO:** Selección de mejores detecciones por estabilidad
    ✅ **NUEVO:** Soporte para archivos hasta 150MB

    ### 📊 Mejoras en Tracking:
    - **Doble Confianza**: Maneja independientemente la confianza del detector de placas y del reconocedor de caracteres
    - **Tracking Estable**: Requiere múltiples detecciones consistentes antes de confirmar una placa
    - **Anti-Duplicación**: Evita reconocer la misma placa múltiples veces en el mismo vehículo
    - **Calidad Temporal**: Evalúa la estabilidad de las detecciones a lo largo del tiempo
    - **Archivos Grandes**: Procesa videos de hasta 150MB y 10 minutos de duración
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 🔧 CONFIGURAR MIDDLEWARES PARA ARCHIVOS GRANDES

# 1. Middleware personalizado para archivos grandes
app.add_middleware(LargeFileMiddleware)

# 2. CORS con configuración extendida
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight por 1 hora
)

# 3. Trusted Host (opcional para producción)
# app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

# Montar archivos estáticos
try:
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    logger.info(f"📁 Archivos estáticos montados en: /static -> {settings.static_dir}")
except Exception as e:
    logger.warning(f"⚠️ No se pudieron montar archivos estáticos: {str(e)}")

# Importar después de configurar logging
from models.model_manager import model_manager
from api.routes import health, detection
from api.routes.video import video_router

# Incluir rutas
app.include_router(health.router)
app.include_router(detection.router)
app.include_router(video_router)

logger.info("🛣️ Rutas registradas:")
logger.info("   📊 Health: /api/v1/health/*")
logger.info("   🔍 Detection: /api/v1/detect/*")
logger.info("   🎬 Video: /api/v1/video/*")


# Endpoint raíz actualizado
@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz de la API con información completa"""

    # Obtener estado de los modelos
    try:
        models_status = model_manager.get_model_info() if model_manager.is_loaded else {"models_loaded": False}
    except:
        models_status = {"models_loaded": False}

    return {
        "message": "🚗 CARID - Sistema ALPR",
        "version": settings.app_version,
        "status": "running",
        "etapa_actual": "3+ - Tracking Avanzado con Doble Confianza",
        "funcionalidades": {
            "deteccion_placas_imagenes": "✅ Disponible",
            "reconocimiento_caracteres": "✅ Disponible",
            "validacion_formato": "✅ Disponible",
            "procesamiento_videos": "✅ Disponible",
            "tracking_avanzado": "✅ **NUEVO** Doble confianza + estabilidad",
            "anti_duplicacion": "✅ **MEJORADO** Sistema inteligente",
            "archivos_grandes": "✅ **NUEVO** Hasta 150MB",
            "streaming_tiempo_real": "⏳ Próximamente"
        },
        "modelos": {
            "cargados": models_status.get("models_loaded", False),
            "dispositivo": models_status.get("device", "unknown"),
            "detector_placas": models_status.get("plate_detector_loaded", False),
            "reconocedor_caracteres": models_status.get("char_recognizer_loaded", False)
        },
        "endpoints": {
            "deteccion_imagen": "/api/v1/detect/image",
            "deteccion_rapida": "/api/v1/detect/image/quick",
            "deteccion_video": "/api/v1/video/detect",
            "deteccion_video_rapida": "/api/v1/video/detect/quick",
            "estadisticas_video": "/api/v1/video/stats",
            "health_check": "/api/v1/health",
            "documentacion": "/docs",
            "estadisticas": "/api/v1/detect/stats"
        },
        "configuracion": {
            "max_file_size_mb": settings.max_file_size,  # ✅ ACTUALIZADO A 150MB
            "formatos_imagenes": ["jpg", "jpeg", "png"],
            "formatos_videos": ["mp4", "avi", "mov", "mkv", "webm"],
            "max_video_duration": settings.max_video_duration,  # ✅ 10 minutos
            "cuda_disponible": models_status.get("cuda_available", False)
        },
        "novedades_tracking_avanzado": {
            "doble_confianza": "Maneja independientemente detector y reconocedor",
            "estabilidad_temporal": "Requiere múltiples detecciones consistentes",
            "anti_duplicacion_inteligente": "Evita re-detectar la misma placa",
            "calidad_tracking": "Evalúa excellent/good/fair/poor",
            "archivos_grandes": "Hasta 150MB y 10 minutos",
            "pesos_configurables": "Detector 40% + Reconocedor 60%",
            "iou_tracking": "Seguimiento espacial mejorado",
            "limpieza_automatica": "Trackers inactivos se archivan automáticamente"
        }
    }


# Endpoint de información del sistema (sin cambios)
@app.get("/system", tags=["System"])
async def system_info():
    """Información detallada del sistema"""
    try:
        from api.dependencies import get_system_info
        return get_system_info()
    except Exception as e:
        logger.error(f"❌ Error obteniendo info del sistema: {str(e)}")
        return {"error": "No se pudo obtener información del sistema"}


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
            "help": "Contacte al administrador si el problema persiste"
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
    # Mensaje de inicio actualizado
    logger.info("🚗 CARID ALPR - Etapa 3+: Tracking Avanzado con Doble Confianza")
    logger.info("=" * 70)
    logger.info("🎯 Funcionalidades disponibles:")
    logger.info("   ✅ Detección de placas en imágenes")
    logger.info("   ✅ Reconocimiento de caracteres")
    logger.info("   ✅ Validación de formatos peruanos")
    logger.info("   ✅ Procesamiento de videos")
    logger.info("   ✅ Tracking inteligente con doble confianza")
    logger.info("   ✅ Sistema anti-duplicación avanzado")
    logger.info("   ✅ Evaluación de estabilidad temporal")
    logger.info("   ✅ Soporte para archivos grandes (150MB)")
    logger.info("   ✅ API REST completa")
    logger.info("   ✅ Documentación interactiva")
    logger.info("=" * 70)
    logger.info("🎬 Capacidades de video mejoradas:")
    logger.info("   📹 MP4, AVI, MOV, MKV, WebM hasta 150MB")
    logger.info("   ⏱️ Duración máxima: 10 minutos")
    logger.info("   🎯 Doble confianza: Detector + Reconocedor")
    logger.info("   🔄 Tracking estable con múltiples validaciones")
    logger.info("   🚀 Procesamiento optimizado y paralelo")
    logger.info("=" * 70)

    # Ejecutar servidor con configuración para archivos grandes
    logger.info("🚀 Iniciando servidor uvicorn con soporte para archivos grandes...")

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
        # 🔧 CONFIGURACIONES PARA ARCHIVOS GRANDES
        timeout_keep_alive=30,  # Mantener conexiones por más tiempo
        limit_max_requests=1000,  # Máximo requests por worker
        backlog=2048  # Backlog de conexiones
    )