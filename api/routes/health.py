from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from datetime import datetime
import time

from api.dependencies import (
    get_model_manager,
    get_settings,
    validate_service_health,
    get_system_info
)
from models.model_manager import ModelManager
from config.settings import Settings

router = APIRouter(prefix="/api/v1", tags=["Health Check"])


@router.get("/health", summary="Health Check Básico")
async def health_check():
    """Health check básico que siempre responde si la API está corriendo"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "CARID-ALPR-API",
        "version": "1.0.0"
    }


@router.get("/health/detailed", summary="Health Check Detallado")
async def detailed_health_check(
        models: ModelManager = Depends(get_model_manager),
        settings: Settings = Depends(get_settings),
        _: bool = Depends(validate_service_health)
):
    """Health check detallado que verifica todos los componentes"""

    start_time = time.time()

    # Información de modelos
    model_info = models.get_model_info()

    # Validación de archivos
    file_validation = settings.validate_model_files()

    # Calcular tiempo de respuesta
    response_time = time.time() - start_time

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.app_name,
        "version": settings.app_version,
        "response_time_seconds": round(response_time, 3),
        "models_trained": model_info,
        "files": file_validation,
        "configuration": {
            "device": settings.device,
            "cuda_available": settings.is_cuda_available,
            "confidence_threshold": settings.model_confidence_threshold,
            "iou_threshold": settings.model_iou_threshold,
            "max_file_size_mb": settings.max_file_size,
            "allowed_extensions": settings.allowed_extensions_list
        }
    }


@router.get("/models/info", summary="Información de Modelos")
async def models_info(models: ModelManager = Depends(get_model_manager)):
    """Retorna información detallada sobre los modelos cargados"""
    return models.get_model_info()


@router.get("/system/info", summary="Información del Sistema")
async def system_info():
    """Retorna información del sistema para diagnósticos"""
    return get_system_info()


@router.post("/models/warmup", summary="Warmup de Modelos")
async def warmup_models(models: ModelManager = Depends(get_model_manager)):
    """Realiza warmup de los modelos para optimizar el primer uso"""
    start_time = time.time()

    try:
        models.warmup_models()
        warmup_time = time.time() - start_time

        return {
            "status": "success",
            "message": "Warmup completado exitosamente",
            "warmup_time_seconds": round(warmup_time, 3),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante warmup: {str(e)}"
        )


@router.get("/health/ready", summary="Readiness Check")
async def readiness_check(
        _: bool = Depends(validate_service_health)
):
    """Verifica si el servicio está listo para recibir requests"""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "El servicio está listo para procesar requests"
    }