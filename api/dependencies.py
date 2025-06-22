from fastapi import Depends, HTTPException, status
from config.settings import settings
from loguru import logger
from typing import Dict, Any


async def get_settings():
    """Dependencia para obtener la configuración"""
    return settings


async def get_model_manager():
    """Dependencia para obtener el gestor de modelos"""
    # Importación diferida para evitar importaciones circulares
    from models.model_manager import model_manager

    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Los modelos no están cargados. El servicio no está disponible."
        )
    return model_manager


async def validate_service_health():
    """Dependencia para validar que el servicio esté funcionando correctamente"""
    # Importación diferida para evitar importaciones circulares
    from models.model_manager import model_manager

    health_issues = []

    # Verificar que los modelos estén cargados
    if not model_manager.is_loaded:
        health_issues.append("Modelos no cargados")

    # Verificar que los directorios existan
    try:
        settings.create_directories()
    except Exception as e:
        health_issues.append(f"Error en directorios: {str(e)}")

    # Verificar archivos de modelos
    validation = settings.validate_model_files()
    if not validation["plate_model_exists"]:
        health_issues.append("Archivo de modelo de placas no encontrado")
    if not validation["char_model_exists"]:
        health_issues.append("Archivo de modelo de caracteres no encontrado")

    if health_issues:
        logger.warning(f"Problemas de salud del servicio: {health_issues}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "El servicio tiene problemas de salud",
                "issues": health_issues
            }
        )

    return True


def get_system_info() -> Dict[str, Any]:
    """Obtiene información del sistema para diagnósticos"""
    import torch
    import cv2
    import platform
    import psutil
    from ultralytics import __version__ as ultralytics_version

    system_info = {
        "platform": {
            "system": platform.system(),
            "version": platform.version(),
            "architecture": platform.architecture(),
            "processor": platform.processor()
        },
        "python": {
            "version": platform.python_version()
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
            "percent_used": psutil.virtual_memory().percent
        },
        "libraries": {
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "ultralytics": ultralytics_version
        },
        "gpu": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    }

    # Información adicional de GPU si está disponible
    if torch.cuda.is_available():
        try:
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "device_id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "multi_processor_count": props.multi_processor_count
                })
            system_info["gpu"]["devices"] = gpu_info
        except Exception as e:
            logger.warning(f"Error obteniendo info GPU: {str(e)}")

    return system_info


async def log_request_info(request_id: str = None):
    """Dependencia para logging de requests"""
    if not request_id:
        import uuid
        request_id = str(uuid.uuid4())[:8]

    logger.info(f"🔍 Request ID: {request_id}")
    return request_id