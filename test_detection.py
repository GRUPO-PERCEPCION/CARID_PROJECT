#!/usr/bin/env python3
"""
Script de prueba para la Etapa 2: Funcionalidad de Imágenes
Prueba todos los endpoints de detección implementados
"""

import requests
import json
import time
import os
from pathlib import Path
from loguru import logger


class DetectionTester:
    """Clase para testing de la funcionalidad de detección"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

        # Configurar logging
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n"
        )

    def test_health_check(self):
        """Prueba health check básico"""
        logger.info("🏥 Probando health check...")

        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")

            if response.status_code == 200:
                data = response.json()
                logger.success(f"✅ Health check OK: {data['status']}")
                return True
            else:
                logger.error(f"❌ Health check falló: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en health check: {str(e)}")
            return False

    def test_detailed_health(self):
        """Prueba health check detallado"""
        logger.info("🔍 Probando health check detallado...")

        try:
            response = self.session.get(f"{self.base_url}/api/v1/health/detailed")

            if response.status_code == 200:
                data = response.json()
                logger.success("✅ Health check detallado OK")

                # Mostrar información relevante
                config = data.get("configuration", {})
                logger.info(f"   📱 Dispositivo: {config.get('device', 'unknown')}")
                logger.info(f"   🚀 CUDA: {'Sí' if config.get('cuda_available') else 'No'}")
                logger.info(f"   🎯 Umbral confianza: {config.get('confidence_threshold', 'N/A')}")

                models = data.get("models_trained", {})
                logger.info(f"   🤖 Modelos cargados: {'Sí' if models.get('models_loaded') else 'No'}")

                return True
            else:
                logger.error(f"❌ Health check detallado falló: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en health check detallado: {str(e)}")
            return False

    def test_models_info(self):
        """Prueba endpoint de información de modelos"""
        logger.info("🤖 Probando información de modelos...")

        try:
            response = self.session.get(f"{self.base_url}/api/v1/models/info")

            if response.status_code == 200:
                data = response.json()
                logger.success("✅ Información de modelos OK")

                logger.info(f"   📦 Detector placas: {'✅' if data.get('plate_detector_loaded') else '❌'}")
                logger.info(f"   📖 Reconocedor chars: {'✅' if data.get('char_recognizer_loaded') else '❌'}")
                logger.info(f"   💾 Dispositivo: {data.get('device', 'unknown')}")

                return True
            else:
                logger.error(f"❌ Info de modelos falló: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en info de modelos: {str(e)}")
            return False

    def test_warmup(self):
        """Prueba warmup de modelos"""
        logger.info("🔥 Probando warmup de modelos...")

        try:
            response = self.session.post(f"{self.base_url}/api/v1/models/warmup")

            if response.status_code == 200:
                data = response.json()
                warmup_time = data.get("warmup_time_seconds", 0)
                logger.success(f"✅ Warmup completado en {warmup_time}s")
                return True
            else:
                logger.error(f"❌ Warmup falló: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en warmup: {str(e)}")
            return False

    def test_detection_stats(self):
        """Prueba endpoint de estadísticas"""
        logger.info("📊 Probando estadísticas de detección...")

        try:
            response = self.session.get(f"{self.base_url}/api/v1/detect/stats")

            if response.status_code == 200:
                data = response.json()["data"]
                logger.success("✅ Estadísticas OK")

                models_status = data.get("models_status", {})
                file_system = data.get("file_system", {})

                logger.info(f"   🤖 Modelos cargados: {'✅' if models_status.get('loaded') else '❌'}")
                logger.info(f"   📁 Archivos temp: {file_system.get('temp_files_count', 0)}")
                logger.info(f"   📁 Archivos resultados: {file_system.get('result_files_count', 0)}")

                return True
            else:
                logger.error(f"❌ Estadísticas fallaron: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en estadísticas: {str(e)}")
            return False

    def test_validate_params(self):
        """Prueba validación de parámetros"""
        logger.info("✅ Probando validación de parámetros...")

        # Parámetros válidos
        valid_params = {
            "confidence_threshold": 0.5,
            "iou_threshold": 0.4,
            "max_detections": 5,
            "enhance_image": False,
            "return_visualization": True,
            "save_results": True
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/detect/validate-params",
                json=valid_params
            )

            if response.status_code == 200:
                data = response.json()
                if data["is_valid"]:
                    logger.success("✅ Validación de parámetros OK")
                    if data["warnings"]:
                        logger.info(f"   ⚠️ Advertencias: {data['warnings']}")
                    return True
                else:
                    logger.error(f"❌ Parámetros inválidos: {data['errors']}")
                    return False
            else:
                logger.error(f"❌ Validación falló: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en validación: {str(e)}")
            return False

    def create_test_image(self):
        """Crea una imagen de prueba simple"""
        try:
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont

            # Crear imagen de 640x480 con fondo azul
            img = Image.new('RGB', (640, 480), color='lightblue')
            draw = ImageDraw.Draw(img)

            # Dibujar un rectángulo simulando una placa
            plate_coords = [(200, 200), (440, 260)]
            draw.rectangle(plate_coords, fill='white', outline='black', width=3)

            # Intentar agregar texto (sin fuente específica)
            try:
                draw.text((220, 220), "ABC-123", fill='black')
            except:
                # Si no se puede cargar fuente, solo crear rectángulo
                pass

            # Guardar imagen temporal
            test_image_path = "test_plate_image.jpg"
            img.save(test_image_path, quality=95)

            logger.info(f"📷 Imagen de prueba creada: {test_image_path}")
            return test_image_path

        except Exception as e:
            logger.warning(f"⚠️ No se pudo crear imagen de prueba: {str(e)}")
            return None

    def test_image_detection(self, image_path: str = None):
        """Prueba detección en imagen"""
        logger.info("🔍 Probando detección en imagen...")

        # Crear imagen de prueba si no se proporciona una
        if not image_path:
            image_path = self.create_test_image()
            if not image_path:
                logger.error("❌ No se puede probar sin imagen")
                return False

        if not os.path.exists(image_path):
            logger.error(f"❌ Archivo no encontrado: {image_path}")
            return False

        try:
            # Preparar archivos y datos
            files = {
                'file': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
            }

            data = {
                'confidence_threshold': 0.5,
                'iou_threshold': 0.4,
                'max_detections': 5,
                'return_visualization': True,
                'save_results': True
            }

            logger.info("📤 Enviando imagen para detección...")
            start_time = time.time()

            response = self.session.post(
                f"{self.base_url}/api/v1/detect/image",
                files=files,
                data=data
            )

            processing_time = time.time() - start_time

            # Cerrar archivo
            files['file'][1].close()

            if response.status_code == 200:
                result = response.json()
                logger.success(f"✅ Detección completada en {processing_time:.3f}s")

                # Analizar resultados
                data = result["data"]
                logger.info(f"   🎯 Éxito: {'✅' if data['success'] else '❌'}")
                logger.info(f"   📊 Placas procesadas: {data['plates_processed']}")
                logger.info(f"   ⏱️ Tiempo procesamiento: {data['processing_time']}s")

                if data["best_result"]:
                    best = data["best_result"]
                    logger.info(f"   🏆 Mejor resultado: '{best['plate_text']}'")
                    logger.info(f"   🎯 Confianza: {best['overall_confidence']:.3f}")
                    logger.info(f"   ✅ Formato válido: {'Sí' if best['is_valid_plate'] else 'No'}")
                else:
                    logger.info("   📭 No se detectaron placas")

                # URLs de resultados
                if data.get("result_urls"):
                    logger.info("   🔗 URLs de resultados:")
                    for key, url in data["result_urls"].items():
                        logger.info(f"      {key}: {url}")

                return True
            else:
                logger.error(f"❌ Detección falló: {response.status_code}")
                logger.error(f"   Detalle: {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en detección: {str(e)}")
            return False
        finally:
            # Limpiar imagen de prueba creada
            if not image_path or image_path == "test_plate_image.jpg":
                try:
                    os.remove("test_plate_image.jpg")
                except:
                    pass

    def test_quick_detection(self, image_path: str = None):
        """Prueba detección rápida"""
        logger.info("⚡ Probando detección rápida...")

        # Crear imagen de prueba si no se proporciona
        if not image_path:
            image_path = self.create_test_image()
            if not image_path:
                logger.error("❌ No se puede probar sin imagen")
                return False

        if not os.path.exists(image_path):
            logger.error(f"❌ Archivo no encontrado: {image_path}")
            return False

        try:
            files = {
                'file': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
            }

            data = {
                'confidence_threshold': 0.6
            }

            start_time = time.time()

            response = self.session.post(
                f"{self.base_url}/api/v1/detect/image/quick",
                files=files,
                data=data
            )

            processing_time = time.time() - start_time

            # Cerrar archivo
            files['file'][1].close()

            if response.status_code == 200:
                result = response.json()
                logger.success(f"✅ Detección rápida completada en {processing_time:.3f}s")

                logger.info(f"   🎯 Éxito: {'✅' if result['success'] else '❌'}")
                logger.info(f"   📋 Texto: '{result['plate_text']}'")
                logger.info(f"   🎯 Confianza: {result['confidence']:.3f}")
                logger.info(f"   ✅ Válido: {'Sí' if result['is_valid_format'] else 'No'}")
                logger.info(f"   ⏱️ Tiempo: {result['processing_time']}s")

                return True
            else:
                logger.error(f"❌ Detección rápida falló: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en detección rápida: {str(e)}")
            return False
        finally:
            # Limpiar imagen de prueba
            if not image_path or image_path == "test_plate_image.jpg":
                try:
                    os.remove("test_plate_image.jpg")
                except:
                    pass

    def test_cleanup(self):
        """Prueba limpieza de archivos temporales"""
        logger.info("🗑️ Probando limpieza de archivos...")

        try:
            response = self.session.delete(f"{self.base_url}/api/v1/detect/cleanup?max_age_hours=0")

            if response.status_code == 200:
                logger.success("✅ Limpieza completada")
                return True
            else:
                logger.error(f"❌ Limpieza falló: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en limpieza: {str(e)}")
            return False

    def run_all_tests(self, image_path: str = None):
        """Ejecuta todos los tests"""
        logger.info("🧪 INICIANDO TESTS DE LA ETAPA 2")
        logger.info("=" * 50)

        tests = [
            ("Health Check Básico", self.test_health_check),
            ("Health Check Detallado", self.test_detailed_health),
            ("Información de Modelos", self.test_models_info),
            ("Warmup de Modelos", self.test_warmup),
            ("Estadísticas", self.test_detection_stats),
            ("Validación de Parámetros", self.test_validate_params),
            ("Detección en Imagen", lambda: self.test_image_detection(image_path)),
            ("Detección Rápida", lambda: self.test_quick_detection(image_path)),
            ("Limpieza de Archivos", self.test_cleanup)
        ]

        results = []

        for test_name, test_func in tests:
            logger.info("-" * 30)
            start_time = time.time()

            try:
                success = test_func()
                duration = time.time() - start_time

                results.append({
                    "test": test_name,
                    "success": success,
                    "duration": round(duration, 3)
                })

                if success:
                    logger.success(f"✅ {test_name} - OK ({duration:.3f}s)")
                else:
                    logger.error(f"❌ {test_name} - FALLÓ ({duration:.3f}s)")

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ {test_name} - ERROR: {str(e)}")
                results.append({
                    "test": test_name,
                    "success": False,
                    "duration": round(duration, 3),
                    "error": str(e)
                })

        # Resumen final
        logger.info("=" * 50)
        logger.info("📊 RESUMEN DE TESTS")

        passed = sum(1 for r in results if r["success"])
        total = len(results)
        total_time = sum(r["duration"] for r in results)

        logger.info(f"✅ Tests exitosos: {passed}/{total}")
        logger.info(f"❌ Tests fallidos: {total - passed}/{total}")
        logger.info(f"⏱️ Tiempo total: {total_time:.3f}s")

        if passed == total:
            logger.success("🎉 ¡TODOS LOS TESTS PASARON! ETAPA 2 COMPLETAMENTE FUNCIONAL")
        else:
            logger.warning("⚠️ Algunos tests fallaron. Revisar errores arriba.")

        return results


def main():
    """Función principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Test de la Etapa 2 - Funcionalidad de Imágenes")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base de la API")
    parser.add_argument("--image", help="Ruta de imagen específica para probar")
    parser.add_argument("--test", choices=["health", "detection", "quick", "all"],
                        default="all", help="Tipo de test a ejecutar")

    args = parser.parse_args()

    tester = DetectionTester(args.url)

    logger.info(f"🚀 Probando API en: {args.url}")

    if args.test == "health":
        tester.test_health_check()
        tester.test_detailed_health()
    elif args.test == "detection":
        tester.test_image_detection(args.image)
    elif args.test == "quick":
        tester.test_quick_detection(args.image)
    elif args.test == "all":
        tester.run_all_tests(args.image)


if __name__ == "__main__":
    main()