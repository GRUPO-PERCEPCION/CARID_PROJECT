from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
import sys
import os

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
from api.routes import health


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

    except Exception as e:
        logger.error(f"❌ Error durante startup: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Cerrando CARID ALPR API...")
    logger.info("✅ API cerrada exitosamente")


# Crear aplicación FastAPI
app = FastAPI(
    title="CARID - Sistema ALPR",
    description="""
    ## 🚗 CARID - Sistema de Reconocimiento Automático de Placas Vehiculares

    Sistema avanzado de detección y reconocimiento de placas vehiculares usando YOLOv8.

    ### Características:
    - 🎯 **Detección precisa** de placas vehiculares
    - 📖 **Reconocimiento de caracteres** con alta precisión
    - ⚡ **Procesamiento optimizado** con GPU CUDA
    - 🔧 **API REST** completa y documentada

    ### Tecnologías:
    - **YOLOv8** para detección
    - **PyTorch** con soporte CUDA
    - **FastAPI** para la API REST
    - **OpenCV** para procesamiento de imágenes
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS (abierto para desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(health.router)


# Endpoint raíz
@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz de la API"""
    return {
        "message": "🚗 CARID - Sistema ALPR",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Manejador de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error global: {str(exc)}")
    return {
        "error": "Error interno del servidor",
        "detail": str(exc) if settings.debug else "Contacte al administrador"
    }


if __name__ == "__main__":
    # Ejecutar servidor
    logger.info("🚀 Iniciando servidor uvicorn...")

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )