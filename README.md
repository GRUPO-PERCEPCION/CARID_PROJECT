# 🚗 CARID - Sistema de Identificación de Placas Vehiculares

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Descripción

**CARID** es un sistema avanzado de **Reconocimiento Automático de Placas Vehiculares (ALPR)** desarrollado para el control automático en estacionamientos. El sistema utiliza técnicas de percepción computacional y deep learning para detectar y reconocer placas vehiculares en tiempo real, adaptado específicamente para las condiciones y formatos de placas peruanas.

### ✨ Características Principales

- 🎯 **Detección precisa** de placas vehiculares usando YOLOv8
- 📖 **Reconocimiento de caracteres** con EasyOCR y PaddleOCR
- ⚡ **Procesamiento en tiempo real** (<150ms por imagen)
- 🌐 **Integración con SUNARP** para consultas vehiculares
- 🔧 **Arquitectura modular** y escalable
- 📱 **Compatibilidad multiplataforma**

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|------------|--------|-----------|
| **Python** | 3.9    | Lenguaje principal |
| **YOLOv8** | Latest | Detección de placas |
| **OpenCV** | 4.x    | Procesamiento de imágenes |
| **PyTorch** | Latest | Framework de deep learning |
| **NumPy** | Latest | Cálculos numéricos |
| **Pandas** | Latest | Análisis de datos |

## 🎯 Resultados y Métricas

- ✅ **Precisión de detección**: >90%
- ✅ **Precisión de OCR**: >90%
- ⚡ **Tiempo de inferencia**: <150ms por imagen
- 🔧 **Optimización**: Transfer learning implementado
- 📊 **Hardware probado**: Raspberry Pi 4 + Intel Neural Compute Stick

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Cámara web o acceso a imágenes/videos

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/carid-alpr.git
cd carid-alpr
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Descargar modelos pre-entrenados**
```bash
python download_models.py
```

## 📖 Uso

### Uso Básico

#### Detección en imagen única
```python
from carid import ALPRSystem

# Inicializar el sistema
alpr = ALPRSystem()

# Procesar una imagen
result = alpr.process_image("path/to/image.jpg")
print(f"Placa detectada: {result['plate_text']}")
```

#### Detección en tiempo real (webcam)
```python
from carid import ALPRSystem

# Inicializar sistema con webcam
alpr = ALPRSystem()
alpr.start_real_time_detection()
```

#### Procesamiento de video
```python
from carid import ALPRSystem

alpr = ALPRSystem()
results = alpr.process_video("path/to/video.mp4")
```

### Configuración Avanzada

```python
from carid import ALPRSystem

# Configuración personalizada
config = {
    'confidence_threshold': 0.5,
    'ocr_engine': 'easyocr',  # 'easyocr' o 'paddleocr'
    'gpu_acceleration': True,
    'sunarp_integration': True
}

alpr = ALPRSystem(config=config)
```

## 📁 Estructura del Proyecto

```
CARID/
├── api/                    # API y servicios web
│   ├── __init__.py
│   ├── crud.py            # Operaciones CRUD
│   ├── database.py        # Configuración de BD
│   ├── models.py          # Modelos de datos
│   ├── schemas.py         # Esquemas Pydantic
│   └── websocket.py       # WebSocket para tiempo real
├── models_trained/        # Modelos entrenados
│   └── models_here.txt
├── producer/              # Productor de datos
├── workers/               # Workers para procesamiento
│   ├── __init__.py
│   └── worker_main.py
├── main.py               # Script principal
├── requirements.txt      # Dependencias
├── README.md            # Este archivo
└── LICENSE              # Licencia del proyecto
```

## 🔧 API Endpoints

### Endpoints Principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/detect` | Detectar placa en imagen |
| `GET` | `/api/history` | Historial de detecciones |
| `POST` | `/api/sunarp/query` | Consultar SUNARP |
| `WS` | `/ws/realtime` | WebSocket tiempo real |

### Ejemplo de Uso de API

```bash
# Detectar placa en imagen
curl -X POST "http://localhost:8000/api/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"
```

## 📊 Dataset y Entrenamiento

### Dataset Utilizado
- **Fuente**: [Peru Plate Numbers Dataset - RoboFlow](https://universe.roboflow.com/grupo-6-placas/peru-plate-numbers)
- **Formato**: Anotaciones en formato YOLO
- **Cantidad**: Imágenes variadas de placas peruanas

### Entrenamiento del Modelo
```bash
# Entrenar YOLOv8 personalizado
python train_yolo.py --epochs 200 --batch 8 --imgsz 1024
```

**Hiperparámetros de entrenamiento:**
- `epochs=200`
- `batch=8` 
- `optimizer='AdamW'`
- `imgsz=1024`
- `patience=30`

## 🔍 Integración con SUNARP

El sistema incluye integración experimental con el sistema de consulta vehicular de SUNARP:

```python
from carid.sunarp import SunarpConnector

connector = SunarpConnector()
vehicle_info = connector.query_plate("ABC-123")
print(vehicle_info)
```

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📝 Changelog

### v1.0.0 (2025)
- ✅ Implementación inicial de YOLOv8 para detección
- ✅ Integración de EasyOCR y PaddleOCR
- ✅ API REST funcional
- ✅ WebSocket para tiempo real
- ✅ Integración experimental con SUNARP

## ⚖️ Consideraciones Éticas

Este proyecto ha sido desarrollado considerando:

- **Privacidad**: Procesamiento local de imágenes
- **Sesgos**: Dataset balanceado para diferentes tipos de placas
- **Transparencia**: Código abierto y documentado
- **Uso responsable**: Orientado a aplicaciones de seguridad y control vehicular

## 🐛 Problemas Conocidos y Soluciones

| Problema | Solución |
|----------|----------|
| OpenCV no detecta cámara en Linux | Reinstalar drivers: `sudo apt-get install v4l-utils` |
| Baja precisión con placas reflectantes | Usar filtros polarizadores |
| Dataset desbalanceado | Aumento de datos sintéticos implementado |

## 🔗 Enlaces Útiles

- [Notebook de Entrenamiento YOLOv8](https://colab.research.google.com/drive/1m1NomSMhshiw2ujXQkeODX7yRruVMW4H)
- [Video Demo](https://drive.google.com/file/d/1QRIfRI2TrKORMAkmXkV5H0QliH159p2V/view)
- [Dataset en RoboFlow](https://universe.roboflow.com/grupo-6-placas/peru-plate-numbers)
- [SUNARP Consulta Vehicular](https://consultavehicular.sunarp.gob.pe/consulta-vehicular/inicio)

## 👥 Equipo de Desarrollo

**Universidad Privada Antenor Orrego - Facultad de Ingeniería**
**Escuela de Ingeniería de Sistemas e Inteligencia Artificial**

- **Cisneros Bartra, Adrián**
- **Marin Yupanqui, Bryan** 
- **Mostacero Cieza, Luis**
- **Oncoy Patricio, Ángel**
- **Ulco Lazo, Fabricio**

**Docente:** Armando Javier Caballero Alvarado  
**Curso:** Percepción Computacional  
**NRC:** 9634-9635

## 📜 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 📞 Contacto

Para preguntas, sugerencias o colaboraciones:

- 📧 Email: [contacto@proyecto-carid.com](mailto:contacto@proyecto-carid.com)
- 🐛 Issues: [GitHub Issues](https://github.com/tu-usuario/carid-alpr/issues)
- 💬 Discusiones: [GitHub Discussions](https://github.com/tu-usuario/carid-alpr/discussions)

---

⭐ **¡Si te gusta este proyecto, dale una estrella en GitHub!**