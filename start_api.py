#!/usr/bin/env python3
"""
Script de inicio para CARID ALPR API
Realiza verificaciones previas antes de iniciar el servidor
"""

import os
import sys
import torch
from pathlib import Path
from loguru import logger


def check_environment():
    """Verifica el entorno antes de iniciar"""
    logger.info("🔍 Verificando entorno...")

    # Verificar Python version
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 8:
        logger.error("❌ Python 3.8+ requerido")
        return False

    logger.success(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Verificar archivo .env
    if not os.path.exists(".env"):
        logger.warning("⚠️ Archivo .env no encontrado, usando valores por defecto")
        if os.path.exists(".env.example"):
            logger.info("💡 Considera copiar .env.example a .env y ajustar las configuraciones")
    else:
        logger.success("✅ Archivo .env encontrado")

    # Verificar CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.success(f"✅ CUDA disponible: {gpu_count} GPU(s) - {gpu_name}")
    else:
        logger.warning("⚠️ CUDA no disponible, usando CPU")

    return True


def check_models():
    """Verifica que los modelos existan"""
    logger.info("🤖 Verificando modelos...")

    models_dir = Path("./models_trained")
    if not models_dir.exists():
        logger.error("❌ Directorio ./models_trained no encontrado")
        logger.info("💡 Crea el directorio y coloca tus modelos YOLOv8 (.pt)")
        return False

    # Buscar archivos .pt
    model_files = list(models_dir.glob("*.pt"))

    if len(model_files) == 0:
        logger.error("❌ No se encontraron modelos (.pt) en ./models_trained")
        logger.info("💡 Coloca tus modelos YOLOv8 entrenados en ./models_trained/")
        return False

    logger.success(f"✅ {len(model_files)} modelo(s) encontrado(s):")
    for model in model_files:
        size_mb = model.stat().st_size / (1024 * 1024)
        logger.info(f"   📦 {model.name} ({size_mb:.1f} MB)")

    return True


def check_dependencies():
    """Verifica las dependencias principales"""
    logger.info("📦 Verificando dependencias...")

    try:
        import fastapi
        import uvicorn
        import torch
        import ultralytics
        import cv2
        import PIL

        logger.success("✅ Dependencias principales disponibles")
        logger.info(f"   - FastAPI: {fastapi.__version__}")
        logger.info(f"   - PyTorch: {torch.__version__}")
        logger.info(f"   - Ultralytics: {ultralytics.__version__}")
        logger.info(f"   - OpenCV: {cv2.__version__}")

        return True

    except ImportError as e:
        logger.error(f"❌ Dependencia faltante: {e}")
        logger.info("💡 Ejecuta: pip install -r requirements.txt")
        return False


def create_directories():
    """Crea directorios necesarios"""
    logger.info("📁 Creando directorios...")

    directories = [
        "./uploads",
        "./uploads/temp",
        "./static",
        "./static/results",
        "./logs",
        "./models_trained"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    logger.success("✅ Directorios creados")


def main():
    """Función principal de verificación e inicio"""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    logger.info("🚗 CARID ALPR - Verificación de inicio")
    logger.info("=" * 50)

    # Realizar verificaciones
    checks = [
        ("Entorno", check_environment),
        ("Dependencias", check_dependencies),
        ("Modelos", check_models)
    ]

    all_passed = True
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
            logger.error(f"❌ Verificación '{check_name}' falló")
        else:
            logger.success(f"✅ Verificación '{check_name}' exitosa")
        logger.info("-" * 30)

    if not all_passed:
        logger.error("❌ Algunas verificaciones fallaron")
        logger.info("💡 Revisa los errores anteriores antes de continuar")
        sys.exit(1)

    # Crear directorios
    create_directories()

    logger.success("🎉 Todas las verificaciones pasaron exitosamente")
    logger.info("🚀 Iniciando API...")
    logger.info("=" * 50)

    # Importar y ejecutar la API
    try:
        from main import app
        import uvicorn
        from config.settings import settings

        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level="info"
        )

    except KeyboardInterrupt:
        logger.info("🛑 Detenido por usuario")
    except Exception as e:
        logger.error(f"❌ Error iniciando API: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()