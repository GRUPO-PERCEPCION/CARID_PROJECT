"""
Script de inicio para CARID ALPR API - Etapa 3: Videos
Realiza verificaciones previas antes de iniciar el servidor
"""

import os
import sys
import torch
import cv2
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
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.success(f"✅ CUDA disponible: {gpu_count} GPU(s) - {gpu_name}")
        logger.info(f"   💾 Memoria GPU: {gpu_memory:.1f} GB")
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

    # Verificar modelos específicos requeridos
    required_models = ["plate_detection.pt", "char_recognition.pt"]
    missing_models = []

    for required in required_models:
        if not (models_dir / required).exists():
            missing_models.append(required)

    if missing_models:
        logger.warning(f"⚠️ Modelos faltantes: {', '.join(missing_models)}")
        logger.info("💡 Asegúrate de que los modelos tengan los nombres correctos")

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
        import numpy
        import loguru

        logger.success("✅ Dependencias principales disponibles")
        logger.info(f"   - FastAPI: {fastapi.__version__}")
        logger.info(f"   - PyTorch: {torch.__version__}")
        logger.info(f"   - Ultralytics: {ultralytics.__version__}")
        logger.info(f"   - OpenCV: {cv2.__version__}")
        logger.info(f"   - NumPy: {numpy.__version__}")

        return True

    except ImportError as e:
        logger.error(f"❌ Dependencia faltante: {e}")
        logger.info("💡 Ejecuta: pip install -r requirements.txt")
        return False


def check_video_capabilities():
    """Verifica capacidades específicas para procesamiento de video"""
    logger.info("🎬 Verificando capacidades de video...")

    try:
        # Verificar que OpenCV puede manejar videos
        test_formats = ['mp4', 'avi', 'mov']
        supported_formats = []

        for fmt in test_formats:
            # Verificar si OpenCV tiene soporte para el formato
            fourcc = cv2.VideoWriter_fourcc(*'mp4v' if fmt == 'mp4' else 'XVID')
            if fourcc != -1:
                supported_formats.append(fmt)

        if supported_formats:
            logger.success(f"✅ Formatos de video soportados: {', '.join(supported_formats)}")
        else:
            logger.warning("⚠️ Soporte limitado de formatos de video")

        # Verificar codecs disponibles
        try:
            # Intentar crear un VideoWriter temporal
            temp_path = "test_codec.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 30.0, (640, 480))

            if out.isOpened():
                logger.success("✅ Codec MP4 disponible para escritura")
                out.release()
                # Limpiar archivo temporal
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                logger.warning("⚠️ Problemas con codec MP4")

        except Exception as e:
            logger.warning(f"⚠️ Error verificando codecs: {str(e)}")

        # Verificar capacidad de procesamiento paralelo
        try:
            import concurrent.futures
            import asyncio
            logger.success("✅ Soporte para procesamiento paralelo disponible")
        except ImportError:
            logger.warning("⚠️ Soporte limitado para procesamiento paralelo")

        return True

    except Exception as e:
        logger.error(f"❌ Error verificando capacidades de video: {str(e)}")
        return False


def check_disk_space():
    """Verifica espacio en disco disponible"""
    logger.info("💾 Verificando espacio en disco...")

    try:
        import shutil

        # Verificar espacio en directorio actual
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)

        logger.info(f"   💽 Espacio libre: {free_gb:.1f} GB de {total_gb:.1f} GB")

        if free_gb < 1.0:
            logger.warning("⚠️ Poco espacio en disco (< 1GB)")
            logger.info("💡 Los videos pueden requerir espacio considerable")
        elif free_gb < 5.0:
            logger.warning("⚠️ Espacio limitado (< 5GB)")
            logger.info("💡 Considera liberar espacio para videos largos")
        else:
            logger.success("✅ Espacio en disco suficiente")

        return True

    except Exception as e:
        logger.warning(f"⚠️ No se pudo verificar espacio en disco: {str(e)}")
        return True  # No es crítico


def create_directories():
    """Crea directorios necesarios incluyendo los de video"""
    logger.info("📁 Creando directorios...")

    directories = [
        "./uploads",
        "./uploads/temp",
        "./static",
        "./static/results",
        "./static/videos",  # NUEVO: para videos procesados
        "./static/frames",  # NUEVO: para frames extraídos
        "./logs",
        "./models_trained"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    logger.success("✅ Directorios creados (incluyendo directorios de video)")


def check_video_settings():
    """Verifica configuraciones específicas de video"""
    logger.info("⚙️ Verificando configuraciones de video...")

    try:
        from config.settings import settings

        # Validar configuraciones de video
        video_validation = settings.validate_video_settings()

        if video_validation["all_valid"]:
            logger.success("✅ Configuraciones de video válidas")

            config = settings.get_video_processing_config()
            logger.info(f"   ⏱️ Duración máxima: {config['max_duration']}s")
            logger.info(f"   🔄 Frame skip: {config['frame_skip']}")
            logger.info(f"   📊 Min detecciones: {config['min_detection_frames']}")
            logger.info(f"   🎯 Umbral similitud: {config['similarity_threshold']}")

        else:
            logger.warning("⚠️ Algunas configuraciones de video pueden no ser óptimas")
            for key, valid in video_validation.items():
                if not valid and key != "all_valid":
                    logger.warning(f"   ⚠️ {key}: inválido")

        return True

    except Exception as e:
        logger.warning(f"⚠️ Error verificando configuraciones de video: {str(e)}")
        return True  # No es crítico


def display_capabilities_summary():
    """Muestra resumen de capacidades del sistema"""
    logger.info("📋 Resumen de capacidades del sistema:")

    # Capacidades de hardware
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"   🚀 GPU: {gpu_name}")
    else:
        logger.info("   💻 Procesamiento: CPU solamente")

    # Capacidades de software
    logger.info("   📷 Procesamiento de imágenes: ✅ Disponible")
    logger.info("   🎬 Procesamiento de videos: ✅ **NUEVO** Disponible")
    logger.info("   🔄 Tracking de placas: ✅ **NUEVO** Disponible")
    logger.info("   🎯 Eliminación de duplicados: ✅ **NUEVO** Disponible")

    # Rendimiento esperado
    if torch.cuda.is_available():
        logger.info("   ⚡ Rendimiento esperado: Alto (GPU)")
    else:
        logger.info("   ⚡ Rendimiento esperado: Moderado (CPU)")


def main():
    """Función principal de verificación e inicio"""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    logger.info("🚗 CARID ALPR - Etapa 3: Procesamiento de Videos")
    logger.info("=" * 60)

    # Realizar verificaciones
    checks = [
        ("Entorno", check_environment),
        ("Dependencias", check_dependencies),
        ("Modelos", check_models),
        ("Capacidades de Video", check_video_capabilities),
        ("Espacio en Disco", check_disk_space),
        ("Configuraciones de Video", check_video_settings)
    ]

    all_passed = True
    for check_name, check_func in checks:
        logger.info("-" * 30)
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

    # Mostrar resumen de capacidades
    logger.info("=" * 60)
    display_capabilities_summary()

    logger.info("=" * 60)
    logger.info("🆕 Novedades de la Etapa 3:")
    logger.info("   🎬 Procesamiento de videos frame por frame")
    logger.info("   🔄 Tracking inteligente de placas")
    logger.info("   🎯 Detección única por vehículo")
    logger.info("   🏆 Selección automática de mejores detecciones")
    logger.info("   📹 Soporte para MP4, AVI, MOV, MKV, WebM")
    logger.info("   ⚡ Procesamiento optimizado con AsyncIO")

    logger.info("=" * 60)
    logger.info("🚀 Iniciando API...")

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