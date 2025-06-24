"""
Script de prueba para la Etapa 3: Funcionalidad de Videos
Prueba todos los endpoints de detección en videos implementados
"""

import requests
import json
import time
import os
import cv2
import numpy as np
from pathlib import Path
from loguru import logger


class VideoTester:
    """Clase para testing de la funcionalidad de detección en videos"""

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

    def create_test_video(self, duration_seconds: int = 10, filename: str = "test_video.mp4") -> str:
        """
        Crea un video de prueba con placas simuladas

        Args:
            duration_seconds: Duración del video en segundos
            filename: Nombre del archivo de video

        Returns:
            Ruta del video creado
        """
        try:
            logger.info(f"🎬 Creando video de prueba: {duration_seconds}s")

            # Configuración del video
            fps = 30
            width, height = 640, 480
            total_frames = duration_seconds * fps

            # Crear video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

            # Placas simuladas que aparecerán en diferentes momentos
            test_plates = [
                {"text": "ABC-123", "start_frame": 0, "end_frame": total_frames // 3, "color": (255, 255, 255)},
                {"text": "XYZ-789", "start_frame": total_frames // 3, "end_frame": 2 * total_frames // 3,
                 "color": (255, 255, 255)},
                {"text": "DEF-456", "start_frame": 2 * total_frames // 3, "end_frame": total_frames,
                 "color": (255, 255, 255)}
            ]

            for frame_num in range(total_frames):
                # Crear frame base (fondo azul)
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = (100, 150, 200)  # Fondo azul claro

                # Determinar qué placa mostrar en este frame
                current_plate = None
                for plate in test_plates:
                    if plate["start_frame"] <= frame_num < plate["end_frame"]:
                        current_plate = plate
                        break

                if current_plate:
                    # Calcular posición de la placa (se mueve de izquierda a derecha)
                    plate_range = current_plate["end_frame"] - current_plate["start_frame"]
                    progress = (frame_num - current_plate["start_frame"]) / plate_range

                    # Posición horizontal de la placa
                    plate_x = int(50 + progress * (width - 250))
                    plate_y = height // 2 - 30

                    # Dibujar rectángulo de la placa
                    cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 200, plate_y + 60),
                                  current_plate["color"], -1)
                    cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 200, plate_y + 60),
                                  (0, 0, 0), 2)

                    # Escribir texto de la placa
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 2

                    # Calcular posición centrada del texto
                    text_size = cv2.getTextSize(current_plate["text"], font, font_scale, thickness)[0]
                    text_x = plate_x + (200 - text_size[0]) // 2
                    text_y = plate_y + (60 + text_size[1]) // 2

                    cv2.putText(frame, current_plate["text"], (text_x, text_y),
                                font, font_scale, (0, 0, 0), thickness)

                # Agregar información del frame (opcional)
                cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Escribir frame al video
                out.write(frame)

                # Mostrar progreso cada 30 frames
                if frame_num % 30 == 0:
                    progress = (frame_num / total_frames) * 100
                    logger.info(f"   📊 Progreso creación: {progress:.1f}%")

            # Finalizar video
            out.release()

            logger.success(f"✅ Video de prueba creado: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Error creando video de prueba: {str(e)}")
            return ""

    def test_video_detection(self, video_path: str = None):
        """Prueba detección completa en video"""
        logger.info("🎬 Probando detección en video...")

        # Crear video de prueba si no se proporciona uno
        if not video_path:
            video_path = self.create_test_video(15, "test_plate_video.mp4")
            if not video_path:
                logger.error("❌ No se puede probar sin video")
                return False

        if not os.path.exists(video_path):
            logger.error(f"❌ Archivo no encontrado: {video_path}")
            return False

        try:
            # Preparar archivos y datos
            files = {
                'file': ('test_video.mp4', open(video_path, 'rb'), 'video/mp4')
            }

            data = {
                'confidence_threshold': 0.3,  # Más permisivo para videos de prueba
                'iou_threshold': 0.4,
                'frame_skip': 3,
                'max_duration': 300,
                'save_results': True,
                'save_best_frames': True,
                'create_annotated_video': False,
                'min_detection_frames': 2
            }

            logger.info("📤 Enviando video para detección...")
            start_time = time.time()

            response = self.session.post(
                f"{self.base_url}/api/v1/video/detect",
                files=files,
                data=data,
                timeout=300  # 5 minutos timeout
            )

            processing_time = time.time() - start_time

            # Cerrar archivo
            files['file'][1].close()

            if response.status_code == 200:
                result = response.json()
                logger.success(f"✅ Detección en video completada en {processing_time:.3f}s")

                # Analizar resultados
                data = result["data"]
                logger.info(f"   🎯 Éxito: {'✅' if data['success'] else '❌'}")
                logger.info(f"   📊 Placas únicas encontradas: {len(data['unique_plates'])}")
                logger.info(f"   ⏱️ Tiempo procesamiento: {data['processing_time']}s")

                # Información del video
                video_info = data.get("video_info", {})
                logger.info(f"   📹 Video: {video_info.get('duration', 0):.1f}s, "
                            f"{video_info.get('total_frames', 0)} frames")

                # Resumen de procesamiento
                summary = data.get("processing_summary", {})
                logger.info(f"   📊 Frames procesados: {summary.get('frames_processed', 0)}")
                logger.info(f"   🎯 Frames con detecciones: {summary.get('frames_with_detections', 0)}")
                logger.info(f"   🔍 Total detecciones: {summary.get('total_detections', 0)}")

                if data["unique_plates"]:
                    logger.info("   🏆 Placas detectadas:")
                    for i, plate in enumerate(data["unique_plates"][:5]):  # Mostrar máximo 5
                        logger.info(f"      {i + 1}. '{plate['plate_text']}' - "
                                    f"Confianza: {plate['best_confidence']:.3f} - "
                                    f"Detecciones: {plate['detection_count']} - "
                                    f"Válida: {'✅' if plate['is_valid_format'] else '❌'}")

                    # Información de la mejor placa
                    best_plate = data["best_plate"]
                    logger.info(f"   🥇 Mejor placa: '{best_plate['plate_text']}'")
                    logger.info(f"      📊 Confianza: {best_plate['best_confidence']:.3f}")
                    logger.info(f"      🔄 Detecciones: {best_plate['detection_count']}")
                    logger.info(f"      🎬 Frame inicial: {best_plate['first_seen_frame']}")
                    logger.info(f"      🎬 Frame final: {best_plate['last_seen_frame']}")
                    logger.info(f"      ✅ Formato válido: {'Sí' if best_plate['is_valid_format'] else 'No'}")
                else:
                    logger.info("   📭 No se detectaron placas")

                # URLs de resultados
                if data.get("result_urls"):
                    logger.info("   🔗 URLs de resultados:")
                    for key, url in data["result_urls"].items():
                        logger.info(f"      {key}: {url}")

                return True
            else:
                logger.error(f"❌ Detección en video falló: {response.status_code}")
                logger.error(f"   Detalle: {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en detección de video: {str(e)}")
            return False
        finally:
            # Limpiar video de prueba creado
            if video_path == "test_plate_video.mp4":
                try:
                    os.remove("test_plate_video.mp4")
                except:
                    pass

    def test_quick_video_detection(self, video_path: str = None):
        """Prueba detección rápida en video"""
        logger.info("⚡ Probando detección rápida en video...")

        # Crear video de prueba más corto
        if not video_path:
            video_path = self.create_test_video(5, "test_quick_video.mp4")
            if not video_path:
                logger.error("❌ No se puede probar sin video")
                return False

        if not os.path.exists(video_path):
            logger.error(f"❌ Archivo no encontrado: {video_path}")
            return False

        try:
            files = {
                'file': ('test_quick_video.mp4', open(video_path, 'rb'), 'video/mp4')
            }

            data = {
                'confidence_threshold': 0.4,
                'frame_skip': 5,  # Más agresivo para velocidad
                'max_duration': 60
            }

            start_time = time.time()

            response = self.session.post(
                f"{self.base_url}/api/v1/video/detect/quick",
                files=files,
                data=data,
                timeout=120  # 2 minutos timeout
            )

            processing_time = time.time() - start_time

            # Cerrar archivo
            files['file'][1].close()

            if response.status_code == 200:
                result = response.json()
                logger.success(f"✅ Detección rápida en video completada en {processing_time:.3f}s")

                logger.info(f"   🎯 Éxito: {'✅' if result['success'] else '❌'}")
                logger.info(f"   📊 Placas únicas: {result['unique_plates_count']}")

                if result['success'] and result['unique_plates_count'] > 0:
                    logger.info(f"   📋 Mejor placa: '{result['best_plate_text']}'")
                    logger.info(f"   🎯 Confianza: {result['best_confidence']:.3f}")
                    logger.info(f"   🔄 Detecciones: {result['detection_count']}")
                    logger.info(f"   ✅ Formato válido: {'Sí' if result['is_valid_format'] else 'No'}")
                    logger.info(f"   📊 Frames procesados: {result['frames_processed']}")
                else:
                    logger.info("   📭 No se detectaron placas")

                logger.info(f"   ⏱️ Tiempo: {result['processing_time']}s")

                return True
            else:
                logger.error(f"❌ Detección rápida en video falló: {response.status_code}")
                logger.error(f"   Detalle: {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en detección rápida de video: {str(e)}")
            return False
        finally:
            # Limpiar video de prueba creado
            if video_path == "test_quick_video.mp4":
                try:
                    os.remove("test_quick_video.mp4")
                except:
                    pass

    def test_video_stats(self):
        """Prueba endpoint de estadísticas de video"""
        logger.info("📊 Probando estadísticas de video...")

        try:
            response = self.session.get(f"{self.base_url}/api/v1/video/stats")

            if response.status_code == 200:
                data = response.json()["data"]
                logger.success("✅ Estadísticas de video OK")

                config = data.get("configuration", {})
                capabilities = data.get("processing_capabilities", {})
                performance = data.get("performance", {})

                logger.info(f"   ⏱️ Duración máxima: {config.get('max_video_duration', 0)}s")
                logger.info(f"   📹 Formatos soportados: {len(config.get('supported_formats', []))}")
                logger.info(f"   🔄 Frame skip por defecto: {config.get('default_frame_skip', 0)}")
                logger.info(f"   🚀 GPU disponible: {'✅' if capabilities.get('gpu_acceleration') else '❌'}")
                logger.info(f"   🔧 Procesamiento paralelo: {'✅' if capabilities.get('parallel_processing') else '❌'}")
                logger.info(f"   ⚡ Velocidad promedio: {performance.get('avg_processing_speed', 'N/A')}")

                return True
            else:
                logger.error(f"❌ Estadísticas de video fallaron: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en estadísticas de video: {str(e)}")
            return False

    def test_video_validate_params(self):
        """Prueba validación de parámetros de video"""
        logger.info("✅ Probando validación de parámetros de video...")

        # Parámetros válidos para video
        valid_params = {
            "confidence_threshold": 0.4,
            "iou_threshold": 0.4,
            "frame_skip": 3,
            "max_duration": 180,
            "min_detection_frames": 2
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/video/validate-params",
                data=valid_params
            )

            if response.status_code == 200:
                data = response.json()
                if data["is_valid"]:
                    logger.success("✅ Validación de parámetros de video OK")
                    if data["warnings"]:
                        logger.info(f"   ⚠️ Advertencias: {data['warnings']}")
                    if data["recommendations"]:
                        logger.info(f"   💡 Recomendaciones: {data['recommendations']}")
                    return True
                else:
                    logger.error(f"❌ Parámetros de video inválidos: {data['errors']}")
                    return False
            else:
                logger.error(f"❌ Validación de video falló: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en validación de video: {str(e)}")
            return False

    def test_video_test_pipeline(self):
        """Prueba el endpoint de test pipeline para videos"""
        logger.info("🧪 Probando test pipeline de video...")

        # Crear video específico para testing
        video_path = self.create_test_video(8, "test_pipeline_video.mp4")
        if not video_path:
            logger.error("❌ No se puede probar sin video")
            return False

        try:
            files = {
                'file': ('test_pipeline_video.mp4', open(video_path, 'rb'), 'video/mp4')
            }

            data = {
                'debug': True
            }

            start_time = time.time()

            response = self.session.post(
                f"{self.base_url}/api/v1/video/test-pipeline",
                files=files,
                data=data,
                timeout=180
            )

            processing_time = time.time() - start_time

            # Cerrar archivo
            files['file'][1].close()

            if response.status_code == 200:
                result = response.json()
                logger.success(f"✅ Test pipeline de video completado en {processing_time:.3f}s")

                logger.info(f"   🎯 Éxito: {'✅' if result['success'] else '❌'}")

                # Análisis de video
                video_analysis = result.get("video_analysis", {})
                logger.info(f"   📊 Placas únicas encontradas: {video_analysis.get('unique_plates_found', 0)}")
                logger.info(f"   🏆 Mejor placa: {video_analysis.get('best_plate', 'N/A')}")

                # Información de procesamiento
                processing_summary = result.get("processing_summary", {})
                logger.info(f"   🎬 Frames procesados: {processing_summary.get('frames_processed', 0)}")
                logger.info(f"   🎯 Frames con detecciones: {processing_summary.get('frames_with_detections', 0)}")

                # Información de video
                video_info = result.get("video_info", {})
                logger.info(f"   📹 Duración: {video_info.get('duration', 0):.1f}s")
                logger.info(f"   📊 Total frames: {video_info.get('total_frames', 0)}")

                # Debug info si está disponible
                if result.get("debug_info"):
                    debug = result["debug_info"]
                    tracking = debug.get("tracking_details", {})
                    logger.info("   🔍 Info de debug:")
                    logger.info(f"      Frame skip usado: {debug.get('frame_skip_used', 'N/A')}")
                    logger.info(f"      Umbral confianza: {debug.get('confidence_threshold', 'N/A')}")
                    logger.info(f"      Total detecciones: {tracking.get('total_detections', 0)}")

                return True
            else:
                logger.error(f"❌ Test pipeline de video falló: {response.status_code}")
                logger.error(f"   Detalle: {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en test pipeline de video: {str(e)}")
            return False
        finally:
            # Limpiar video de prueba
            try:
                os.remove("test_pipeline_video.mp4")
            except:
                pass

    def run_all_video_tests(self, video_path: str = None):
        """Ejecuta todos los tests específicos de video"""
        logger.info("🧪 INICIANDO TESTS DE LA ETAPA 3 - VIDEOS")
        logger.info("=" * 60)

        tests = [
            ("Estadísticas de Video", self.test_video_stats),
            ("Validación de Parámetros de Video", self.test_video_validate_params),
            ("Detección en Video", lambda: self.test_video_detection(video_path)),
            ("Detección Rápida en Video", lambda: self.test_quick_video_detection(video_path)),
            ("Test Pipeline de Video", self.test_video_test_pipeline)
        ]

        results = []

        for test_name, test_func in tests:
            logger.info("-" * 40)
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
        logger.info("=" * 60)
        logger.info("📊 RESUMEN DE TESTS DE VIDEO")

        passed = sum(1 for r in results if r["success"])
        total = len(results)
        total_time = sum(r["duration"] for r in results)

        logger.info(f"✅ Tests exitosos: {passed}/{total}")
        logger.info(f"❌ Tests fallidos: {total - passed}/{total}")
        logger.info(f"⏱️ Tiempo total: {total_time:.3f}s")

        if passed == total:
            logger.success("🎉 ¡TODOS LOS TESTS DE VIDEO PASARON!")
            logger.success("🚀 ETAPA 3 - PROCESAMIENTO DE VIDEOS COMPLETAMENTE FUNCIONAL")
        else:
            logger.warning("⚠️ Algunos tests de video fallaron. Revisar errores arriba.")

        return results


def main():
    """Función principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Test de la Etapa 3 - Funcionalidad de Videos")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base de la API")
    parser.add_argument("--video", help="Ruta de video específica para probar")
    parser.add_argument("--test", choices=["stats", "detection", "quick", "pipeline", "all"],
                        default="all", help="Tipo de test a ejecutar")

    args = parser.parse_args()

    tester = VideoTester(args.url)

    logger.info(f"🚀 Probando API de videos en: {args.url}")

    if args.test == "stats":
        tester.test_video_stats()
    elif args.test == "detection":
        tester.test_video_detection(args.video)
    elif args.test == "quick":
        tester.test_quick_video_detection(args.video)
    elif args.test == "pipeline":
        tester.test_video_test_pipeline()
    elif args.test == "all":
        tester.run_all_video_tests(args.video)


if __name__ == "__main__":
    main()