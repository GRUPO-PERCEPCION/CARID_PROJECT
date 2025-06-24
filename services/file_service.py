import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import aiofiles
from fastapi import UploadFile, HTTPException
from PIL import Image
import cv2
import numpy as np
from loguru import logger

from config.settings import settings
from core.utils import (
    is_allowed_file,
    get_file_size_mb,
    validate_file_size,
    generate_unique_filename,
    get_image_dimensions,
    is_valid_image,
    resize_image_if_needed
)


class FileService:
    """Servicio para manejo de archivos y uploads"""

    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.temp_dir = self.upload_dir / "temp"
        self.static_dir = Path(settings.static_dir)
        self.results_dir = self.static_dir / "results"

        # Asegurar que los directorios existan
        self._ensure_directories()

    def _ensure_directories(self):
        """Asegura que todos los directorios necesarios existan"""
        for directory in [self.upload_dir, self.temp_dir, self.static_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    async def save_upload_file(self, upload_file: UploadFile, prefix: str = "") -> Tuple[str, Dict[str, Any]]:
        """
        Guarda un archivo subido y retorna información del mismo

        Args:
            upload_file: Archivo subido por FastAPI
            prefix: Prefijo para el nombre del archivo

        Returns:
            Tupla (ruta_archivo, info_archivo)

        Raises:
            HTTPException: Si hay errores de validación o guardado
        """
        try:
            # Validar que se recibió un archivo
            if not upload_file.filename:
                raise HTTPException(
                    status_code=400,
                    detail="No se recibió ningún archivo"
                )

            # Validar extensión
            if not is_allowed_file(upload_file.filename):
                allowed_extensions = ", ".join(settings.allowed_extensions_list)
                raise HTTPException(
                    status_code=400,
                    detail=f"Tipo de archivo no permitido. Extensiones permitidas: {allowed_extensions}"
                )

            # Generar nombre único
            unique_filename = generate_unique_filename(upload_file.filename, prefix)
            file_path = self.temp_dir / unique_filename

            logger.info(f"💾 Guardando archivo: {unique_filename}")

            # Guardar archivo temporalmente
            async with aiofiles.open(file_path, 'wb') as f:
                content = await upload_file.read()
                await f.write(content)

            # Validar tamaño
            file_size_mb = get_file_size_mb(str(file_path))
            if not validate_file_size(str(file_path)):
                # Eliminar archivo si es muy grande
                os.remove(file_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"Archivo muy grande. Tamaño máximo: {settings.max_file_size}MB, "
                           f"archivo recibido: {file_size_mb:.1f}MB"
                )

            # Validar que sea una imagen válida
            if not is_valid_image(str(file_path)):
                os.remove(file_path)
                raise HTTPException(
                    status_code=400,
                    detail="El archivo no es una imagen válida o está corrupto"
                )

            # Obtener dimensiones
            dimensions = get_image_dimensions(str(file_path))

            # Redimensionar si es necesario
            resize_image_if_needed(str(file_path))

            # Crear información del archivo
            file_info = {
                "filename": unique_filename,
                "original_filename": upload_file.filename,
                "content_type": upload_file.content_type or "image/unknown",
                "size_bytes": os.path.getsize(file_path),
                "size_mb": round(file_size_mb, 2),
                "dimensions": {"width": dimensions[0], "height": dimensions[1]} if dimensions else None
            }

            logger.success(f"✅ Archivo guardado: {unique_filename} ({file_info['size_mb']}MB)")

            return str(file_path), file_info

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error guardando archivo: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error interno guardando archivo: {str(e)}"
            )

    async def save_result_image(self, image: np.ndarray, result_id: str, suffix: str = "result") -> str:
        """
        Guarda una imagen de resultado procesada

        Args:
            image: Imagen como numpy array (RGB)
            result_id: ID único del resultado
            suffix: Sufijo para el nombre del archivo

        Returns:
            Ruta del archivo guardado
        """
        try:
            # Generar nombre de archivo
            filename = f"{result_id}_{suffix}.jpg"
            file_path = self.results_dir / filename

            # Convertir de RGB a BGR para OpenCV
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            # Guardar imagen
            success = cv2.imwrite(str(file_path), image_bgr)

            if not success:
                raise Exception("Error guardando imagen con OpenCV")

            logger.info(f"💾 Imagen resultado guardada: {filename}")
            return str(file_path)

        except Exception as e:
            logger.error(f"❌ Error guardando imagen resultado: {str(e)}")
            raise

    def cleanup_temp_file(self, file_path: str):
        """Elimina un archivo temporal"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🗑️ Archivo temporal eliminado: {Path(file_path).name}")
        except Exception as e:
            logger.warning(f"⚠️ Error eliminando archivo temporal: {str(e)}")

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Limpia archivos antiguos de directorios temporales"""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            # Limpiar archivos temporales
            for directory in [self.temp_dir, self.results_dir]:
                if not directory.exists():
                    continue

                for file_path in directory.iterdir():
                    if file_path.is_file():
                        try:
                            file_age = current_time - file_path.stat().st_ctime
                            if file_age > max_age_seconds:
                                file_path.unlink()
                                logger.info(f"🗑️ Archivo antiguo eliminado: {file_path.name}")
                        except Exception as e:
                            logger.warning(f"⚠️ Error eliminando {file_path.name}: {str(e)}")

        except Exception as e:
            logger.warning(f"⚠️ Error en limpieza de archivos: {str(e)}")

    def get_file_url(self, file_path: str, endpoint_base: str = "http://localhost:8000") -> str:
        """
        Genera URL para acceso a archivo

        Args:
            file_path: Ruta del archivo
            endpoint_base: URL base del endpoint

        Returns:
            URL completa para acceder al archivo
        """
        try:
            # Convertir ruta absoluta a relativa desde static
            path_obj = Path(file_path)

            if self.static_dir in path_obj.parents or path_obj.parent == self.static_dir:
                # Archivo está en static, generar URL relativa
                relative_path = path_obj.relative_to(self.static_dir)
                return f"{endpoint_base}/static/{relative_path}"
            else:
                # Archivo temporal, no generar URL pública
                return ""

        except Exception as e:
            logger.warning(f"⚠️ Error generando URL: {str(e)}")
            return ""

    def create_result_id(self) -> str:
        """Genera un ID único para un resultado"""
        return str(uuid.uuid4())[:12]  # ID corto pero único

    async def copy_to_results(self, source_path: str, result_id: str, suffix: str = "original") -> str:
        """
        Copia un archivo a la carpeta de resultados

        Args:
            source_path: Ruta del archivo origen
            result_id: ID del resultado
            suffix: Sufijo para el nombre

        Returns:
            Ruta del archivo copiado
        """
        try:
            source = Path(source_path)
            extension = source.suffix

            dest_filename = f"{result_id}_{suffix}{extension}"
            dest_path = self.results_dir / dest_filename

            # Copiar archivo
            shutil.copy2(source_path, dest_path)

            logger.info(f"📋 Archivo copiado a resultados: {dest_filename}")
            return str(dest_path)

        except Exception as e:
            logger.error(f"❌ Error copiando archivo: {str(e)}")
            raise


# Instancia global del servicio
file_service = FileService()