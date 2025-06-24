from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger
from config.settings import settings


class BaseModel(ABC):
    """Clase base para todos los modelos YOLO del sistema"""

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.model: Optional[YOLO] = None
        self.device = settings.device
        self.is_loaded = False

    def load_model(self) -> bool:
        """Carga el modelo desde el archivo"""
        try:
            logger.info(f"📦 Cargando {self.model_name}: {self.model_path}")

            # Verificar que el archivo existe
            import os
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Archivo no encontrado: {self.model_path}")
                return False

            # Configurar PyTorch para permitir carga de modelos Ultralytics
            # Solución para PyTorch 2.6+
            try:
                # Método 1: Agregar globals seguros
                torch.serialization.add_safe_globals([
                    'ultralytics.nn.tasks.DetectionModel',
                    'ultralytics.nn.modules.conv.Conv',
                    'ultralytics.nn.modules.block.C2f',
                    'ultralytics.nn.modules.head.Detect',
                    'ultralytics.nn.modules.conv.autopad',
                    'ultralytics.models.yolo.detect.DetectionModel'
                ])
            except Exception as e:
                logger.warning(f"⚠️ No se pudieron agregar globals seguros: {e}")

            # Cargar modelo con configuración especial para PyTorch 2.6+
            logger.info(f"🔧 Cargando modelo con PyTorch {torch.__version__}")

            # Método alternativo: usar weights_only=False temporalmente
            import os
            os.environ['TORCH_SERIALIZATION_SAFE_GLOBALS'] = 'True'

            # Cargar modelo
            self.model = YOLO(self.model_path, task='detect')

            # Mover al dispositivo correcto
            if self.device != "cpu":
                self.model.to(self.device)

            self.is_loaded = True
            logger.success(f"✅ {self.model_name} cargado exitosamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error cargando {self.model_name}: {str(e)}")

            # Intentar método alternativo
            try:
                logger.info(f"🔄 Intentando método alternativo para {self.model_name}...")

                # Método 2: Cargar con contexto seguro
                with torch.serialization.safe_globals([
                    'ultralytics.nn.tasks.DetectionModel',
                    'ultralytics.models.yolo.detect.DetectionModel'
                ]):
                    self.model = YOLO(self.model_path)

                if self.device != "cpu":
                    self.model.to(self.device)

                self.is_loaded = True
                logger.success(f"✅ {self.model_name} cargado con método alternativo")
                return True

            except Exception as e2:
                logger.error(f"❌ Error con método alternativo: {str(e2)}")
                return False

    def predict(self, source, **kwargs) -> List:
        """Realiza predicción con el modelo"""
        if not self.is_loaded or not self.model:
            raise ValueError(f"{self.model_name} no está cargado")

        # Configurar parámetros por defecto (sin verbose duplicado)
        predict_kwargs = {
            'conf': kwargs.get('conf', settings.model_confidence_threshold),
            'iou': kwargs.get('iou', settings.model_iou_threshold),
            'device': self.device
        }

        # Agregar otros kwargs excepto verbose si ya está en predict_kwargs
        for key, value in kwargs.items():
            if key not in predict_kwargs:
                predict_kwargs[key] = value

        try:
            results = self.model.predict(source=source, **predict_kwargs)
            return results

        except Exception as e:
            logger.error(f"❌ Error en predicción {self.model_name}: {str(e)}")
            raise

    @abstractmethod
    def process_results(self, results: List, original_image: np.ndarray) -> Dict[str, Any]:
        """Procesa los resultados del modelo (debe ser implementado por cada subclase)"""
        pass

    def preprocess_image(self, image_input) -> np.ndarray:
        """Preprocesa la imagen antes de la predicción"""
        try:
            # Si es string (path), cargar imagen
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"No se pudo cargar la imagen: {image_input}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Si es numpy array, usar directamente
            elif isinstance(image_input, np.ndarray):
                image = image_input.copy()
                # Convertir de BGR a RGB si es necesario
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            else:
                raise ValueError("Formato de imagen no soportado")

            return image

        except Exception as e:
            logger.error(f"❌ Error preprocesando imagen: {str(e)}")
            raise

    def extract_bboxes(self, results: List) -> List[Dict[str, Any]]:
        """Extrae bounding boxes de los resultados YOLO"""
        detections = []

        try:
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()

                    # Si hay clases disponibles
                    if result.boxes.cls is not None:
                        classes = result.boxes.cls.cpu().numpy()
                        class_names = [result.names[int(cls)] for cls in classes]
                    else:
                        classes = [0] * len(boxes)  # Clase por defecto
                        class_names = ["object"] * len(boxes)

                    for i, (box, conf, cls, cls_name) in enumerate(zip(boxes, confidences, classes, class_names)):
                        detection = {
                            "bbox": box.tolist(),  # [x1, y1, x2, y2]
                            "confidence": float(conf),
                            "class_id": int(cls),
                            "class_name": cls_name,
                            "area": self._calculate_bbox_area(box)
                        }
                        detections.append(detection)

            # Ordenar por confianza (mayor a menor)
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            return detections

        except Exception as e:
            logger.error(f"❌ Error extrayendo bboxes: {str(e)}")
            return []

    def _calculate_bbox_area(self, bbox: np.ndarray) -> float:
        """Calcula el área de un bounding box"""
        x1, y1, x2, y2 = bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return float(width * height)

    def crop_image_from_bbox(self, image: np.ndarray, bbox: List[float], padding: int = 5) -> np.ndarray:
        """Recorta una región de la imagen basada en el bounding box"""
        try:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]

            # Agregar padding y asegurar que esté dentro de los límites
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(w, int(x2) + padding)
            y2 = min(h, int(y2) + padding)

            # Recortar imagen
            cropped = image[y1:y2, x1:x2]

            if cropped.size == 0:
                raise ValueError("El recorte resultó en una imagen vacía")

            return cropped

        except Exception as e:
            logger.error(f"❌ Error recortando imagen: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "confidence_threshold": settings.model_confidence_threshold,
            "iou_threshold": settings.model_iou_threshold
        }

    def warmup(self):
        """Realiza warmup del modelo con una imagen dummy"""
        try:
            if not self.is_loaded:
                logger.warning(f"⚠️ {self.model_name} no está cargado, saltando warmup")
                return

            logger.info(f"🔥 Warmup {self.model_name}...")

            # Crear imagen dummy
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            # Realizar predicción dummy sin verbose para evitar conflictos
            _ = self.predict(dummy_image, verbose=False)

            logger.success(f"✅ Warmup {self.model_name} completado")

        except Exception as e:
            logger.warning(f"⚠️ Error en warmup {self.model_name}: {str(e)}")

    def __str__(self):
        return f"{self.model_name} ({'Cargado' if self.is_loaded else 'No cargado'})"

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.model_name}', loaded={self.is_loaded})>"