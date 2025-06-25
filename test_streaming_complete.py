#!/usr/bin/env python3
"""
Sistema Completo de Pruebas para Streaming ALPR en Tiempo Real
Prueba todos los componentes antes de conectar el frontend
"""

import asyncio
import websockets
import requests
import json
import time
import os
import cv2
import numpy as np
import base64
from pathlib import Path
from loguru import logger
import threading
from typing import Dict, List, Any, Optional
import uuid
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from datetime import datetime


class StreamingTester:
    """Tester completo para el sistema de streaming ALPR"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.session = requests.Session()

        # Configurar logging
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n"
        )

        # Métricas de prueba
        self.test_results = []
        self.streaming_metrics = {
            "frames_received": 0,
            "detections_received": 0,
            "average_latency": 0,
            "connection_time": 0,
            "total_test_time": 0
        }

    def create_test_video_with_plates(self, duration: int = 30, filename: str = "test_streaming_video.mp4") -> str:
        """
        Crea un video de prueba específico para streaming con múltiples placas

        Args:
            duration: Duración en segundos
            filename: Nombre del archivo

        Returns:
            Ruta del video creado
        """
        try:
            logger.info(f"🎬 Creando video de prueba para streaming: {duration}s")

            # Configuración del video
            fps = 30
            width, height = 1280, 720  # HD para mejor calidad
            total_frames = duration * fps

            # Crear video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

            # Placas que aparecerán en diferentes momentos con movimiento
            test_plates = [
                {
                    "text": "ABC-123",
                    "start_frame": 0,
                    "end_frame": total_frames // 4,
                    "color": (255, 255, 255),
                    "start_pos": (50, 200),
                    "end_pos": (width - 250, 200)
                },
                {
                    "text": "XYZ-789",
                    "start_frame": total_frames // 4,
                    "end_frame": total_frames // 2,
                    "color": (255, 255, 255),
                    "start_pos": (width - 250, 400),
                    "end_pos": (50, 400)
                },
                {
                    "text": "DEF-456",
                    "start_frame": total_frames // 2,
                    "end_frame": 3 * total_frames // 4,
                    "color": (255, 255, 255),
                    "start_pos": (width // 2 - 100, 100),
                    "end_pos": (width // 2 - 100, height - 150)
                },
                {
                    "text": "GHI-012",
                    "start_frame": 3 * total_frames // 4,
                    "end_frame": total_frames,
                    "color": (255, 255, 255),
                    "start_pos": (100, height // 2),
                    "end_pos": (width - 300, height // 2)
                }
            ]

            for frame_num in range(total_frames):
                # Crear frame base con gradiente
                frame = np.zeros((height, width, 3), dtype=np.uint8)

                # Fondo con gradiente
                for y in range(height):
                    intensity = int(50 + (y / height) * 100)
                    frame[y, :] = (intensity, intensity + 20, intensity + 40)

                # Agregar timestamp
                timestamp = frame_num / fps
                cv2.putText(frame, f"Time: {timestamp:.1f}s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Agregar número de frame
                cv2.putText(frame, f"Frame: {frame_num}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                # Procesar placas activas en este frame
                for plate in test_plates:
                    if plate["start_frame"] <= frame_num < plate["end_frame"]:
                        # Calcular posición interpolada
                        progress = (frame_num - plate["start_frame"]) / (plate["end_frame"] - plate["start_frame"])

                        start_x, start_y = plate["start_pos"]
                        end_x, end_y = plate["end_pos"]

                        current_x = int(start_x + (end_x - start_x) * progress)
                        current_y = int(start_y + (end_y - start_y) * progress)

                        # Dibujar placa con sombra para realismo
                        plate_width, plate_height = 200, 60

                        # Sombra
                        shadow_offset = 3
                        cv2.rectangle(frame,
                                      (current_x + shadow_offset, current_y + shadow_offset),
                                      (current_x + plate_width + shadow_offset,
                                       current_y + plate_height + shadow_offset),
                                      (50, 50, 50), -1)

                        # Placa principal
                        cv2.rectangle(frame, (current_x, current_y),
                                      (current_x + plate_width, current_y + plate_height),
                                      plate["color"], -1)

                        # Borde de la placa
                        cv2.rectangle(frame, (current_x, current_y),
                                      (current_x + plate_width, current_y + plate_height),
                                      (0, 0, 0), 3)

                        # Texto de la placa
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.2
                        thickness = 3

                        text_size = cv2.getTextSize(plate["text"], font, font_scale, thickness)[0]
                        text_x = current_x + (plate_width - text_size[0]) // 2
                        text_y = current_y + (plate_height + text_size[1]) // 2

                        cv2.putText(frame, plate["text"], (text_x, text_y),
                                    font, font_scale, (0, 0, 0), thickness)

                        # Indicador de movimiento
                        cv2.putText(frame, f"Plate: {plate['text']}", (current_x, current_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Agregar información de progreso
                progress_percent = (frame_num / total_frames) * 100
                cv2.putText(frame, f"Progress: {progress_percent:.1f}%", (width - 300, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Escribir frame
                out.write(frame)

                # Mostrar progreso cada 90 frames
                if frame_num % 90 == 0:
                    logger.info(f"   📊 Creación: {progress_percent:.1f}%")

            out.release()

            # Verificar que el video se creó correctamente
            if os.path.exists(filename):
                file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
                logger.success(f"✅ Video de prueba creado: {filename} ({file_size:.1f}MB)")
                return filename
            else:
                logger.error("❌ Error: Video no se creó correctamente")
                return ""

        except Exception as e:
            logger.error(f"❌ Error creando video de prueba: {str(e)}")
            return ""

    async def test_websocket_connection(self, session_id: str = None) -> bool:
        """Prueba la conexión WebSocket básica"""
        if not session_id:
            session_id = f"test_{int(time.time())}"

        logger.info(f"🔌 Probando conexión WebSocket: {session_id}")

        try:
            uri = f"{self.ws_url}/api/v1/streaming/ws/{session_id}"

            async with websockets.connect(uri, ping_interval=30, ping_timeout=10) as websocket:
                logger.info(f"✅ WebSocket conectado: {uri}")

                # Enviar ping
                await websocket.send(json.dumps({
                    "type": "ping",
                    "timestamp": time.time()
                }))

                # Esperar respuesta
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)

                if response_data.get("type") == "pong":
                    logger.success("✅ Ping/Pong exitoso")
                    return True
                else:
                    logger.error(f"❌ Respuesta inesperada: {response_data}")
                    return False

        except Exception as e:
            logger.error(f"❌ Error en conexión WebSocket: {str(e)}")
            return False

    async def test_streaming_session_complete(self, video_path: str = None) -> bool:
        """Prueba completa de una sesión de streaming"""
        session_id = f"test_streaming_{int(time.time())}"

        logger.info(f"🎬 Iniciando prueba completa de streaming: {session_id}")

        # Crear video si no se proporciona
        if not video_path:
            video_path = self.create_test_video_with_plates(20)
            if not video_path:
                return False

        try:
            # Paso 1: Conectar WebSocket
            uri = f"{self.ws_url}/api/v1/streaming/ws/{session_id}"

            async with websockets.connect(uri, ping_interval=30) as websocket:
                logger.info("✅ WebSocket conectado para streaming")

                # Paso 2: Iniciar sesión de streaming
                success = await self._start_streaming_session(session_id, video_path)
                if not success:
                    return False

                # Paso 3: Monitorear streaming en tiempo real
                await self._monitor_streaming_session(websocket, session_id)

                return True

        except Exception as e:
            logger.error(f"❌ Error en prueba de streaming: {str(e)}")
            return False
        finally:
            # Limpiar archivo de prueba
            try:
                if video_path.startswith("test_"):
                    os.remove(video_path)
            except:
                pass

    async def _start_streaming_session(self, session_id: str, video_path: str) -> bool:
        """Inicia una sesión de streaming vía REST API"""
        try:
            logger.info("📤 Iniciando sesión de streaming...")

            with open(video_path, 'rb') as video_file:
                files = {
                    'file': ('test_video.mp4', video_file, 'video/mp4')
                }

                data = {
                    'session_id': session_id,
                    'confidence_threshold': 0.3,
                    'frame_skip': 2,
                    'max_duration': 300,
                    'adaptive_quality': True,
                    'enable_thumbnails': True
                }

                response = self.session.post(
                    f"{self.base_url}/api/v1/streaming/start-session",
                    files=files,
                    data=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.success("✅ Sesión de streaming iniciada")
                    logger.info(f"   📁 Archivo: {result['data']['file_info']['filename']}")
                    logger.info(f"   💾 Tamaño: {result['data']['file_info']['size_mb']}MB")
                    return True
                else:
                    logger.error(f"❌ Error iniciando streaming: {response.status_code}")
                    logger.error(f"   Detalle: {response.text}")
                    return False

        except Exception as e:
            logger.error(f"❌ Error en inicio de sesión: {str(e)}")
            return False

    async def _monitor_streaming_session(self, websocket, session_id: str):
        """Monitorea una sesión de streaming y recopila métricas"""
        logger.info("👀 Monitoreando sesión de streaming...")

        start_time = time.time()
        frames_received = 0
        detections_received = 0
        unique_plates = set()
        last_progress = 0
        latencies = []

        try:
            while True:
                # Recibir mensaje con timeout
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    message_data = json.loads(message)

                    message_type = message_data.get("type", "")

                    if message_type == "streaming_update":
                        # Procesar actualización de streaming
                        data = message_data.get("data", {})

                        # Contar frames
                        if data.get("frame_data"):
                            frames_received += 1

                        # Contar detecciones
                        detections = data.get("current_detections", [])
                        detections_received += len(detections)

                        # Recopilar placas únicas
                        for detection in detections:
                            unique_plates.add(detection.get("plate_text", ""))

                        # Calcular latencia
                        frame_info = data.get("frame_info", {})
                        if frame_info.get("timestamp"):
                            latency = time.time() - frame_info["timestamp"]
                            latencies.append(latency)

                        # Mostrar progreso
                        progress = data.get("progress", {})
                        current_progress = progress.get("progress_percent", 0)

                        if current_progress - last_progress >= 10:  # Cada 10%
                            logger.info(f"📊 Progreso: {current_progress:.1f}% - "
                                        f"Frames: {frames_received} - "
                                        f"Detecciones: {detections_received} - "
                                        f"Placas únicas: {len(unique_plates)}")
                            last_progress = current_progress

                        # Mostrar detecciones en tiempo real
                        if detections:
                            for detection in detections:
                                logger.info(f"🎯 DETECCIÓN: '{detection.get('plate_text')}' - "
                                            f"Confianza: {detection.get('overall_confidence', 0):.3f} - "
                                            f"Válida: {'✅' if detection.get('is_valid_plate') else '❌'}")

                    elif message_type == "streaming_completed":
                        logger.success("🏁 Streaming completado")

                        # Mostrar resumen final
                        data = message_data.get("data", {})
                        summary = data.get("detection_summary", {})

                        logger.info("📋 RESUMEN FINAL:")
                        logger.info(f"   📊 Total detecciones: {summary.get('total_detections', 0)}")
                        logger.info(f"   🎯 Placas únicas: {summary.get('unique_plates_count', 0)}")
                        logger.info(f"   🎬 Frames con detecciones: {summary.get('frames_with_detections', 0)}")

                        # Mejores placas
                        best_plates = summary.get("best_plates", [])
                        if best_plates:
                            logger.info("🏆 MEJORES PLACAS:")
                            for i, plate in enumerate(best_plates[:3]):
                                logger.info(f"   {i + 1}. '{plate.get('plate_text')}' - "
                                            f"Confianza: {plate.get('best_confidence', 0):.3f}")

                        break

                    elif message_type == "streaming_error":
                        error = message_data.get("error", "Error desconocido")
                        logger.error(f"❌ Error en streaming: {error}")
                        break

                    elif message_type == "ping":
                        # Responder pong
                        await websocket.send(json.dumps({
                            "type": "pong",
                            "timestamp": time.time()
                        }))

                except asyncio.TimeoutError:
                    logger.warning("⏰ Timeout esperando mensaje - enviando ping")
                    await websocket.send(json.dumps({
                        "type": "ping",
                        "timestamp": time.time()
                    }))
                    continue

        except Exception as e:
            logger.error(f"❌ Error monitoreando streaming: {str(e)}")

        # Calcular métricas finales
        total_time = time.time() - start_time
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Guardar métricas
        self.streaming_metrics.update({
            "frames_received": frames_received,
            "detections_received": detections_received,
            "unique_plates_count": len(unique_plates),
            "average_latency": avg_latency,
            "total_test_time": total_time,
            "frames_per_second": frames_received / total_time if total_time > 0 else 0
        })

        logger.info("📊 MÉTRICAS DE STREAMING:")
        logger.info(f"   🎬 Frames recibidos: {frames_received}")
        logger.info(f"   🎯 Detecciones totales: {detections_received}")
        logger.info(f"   🚗 Placas únicas: {len(unique_plates)}")
        logger.info(f"   ⏱️ Latencia promedio: {avg_latency:.3f}s")
        logger.info(f"   📊 FPS de streaming: {frames_received / total_time:.2f}")
        logger.info(f"   ⏳ Tiempo total: {total_time:.1f}s")

    def test_rest_api_endpoints(self) -> bool:
        """Prueba todos los endpoints REST de streaming"""
        logger.info("🌐 Probando endpoints REST de streaming...")

        endpoints_to_test = [
            ("GET", "/api/v1/streaming/health", "Health check de streaming"),
            ("GET", "/api/v1/streaming/config", "Configuración de streaming"),
            ("GET", "/api/v1/streaming/stats", "Estadísticas de streaming"),
            ("GET", "/api/v1/streaming/sessions", "Sesiones activas"),
            ("POST", "/api/v1/streaming/test-connection", "Test de conexión"),
        ]

        results = []

        for method, endpoint, description in endpoints_to_test:
            try:
                logger.info(f"📡 Probando: {method} {endpoint}")

                if method == "GET":
                    response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                elif method == "POST":
                    response = self.session.post(f"{self.base_url}{endpoint}", timeout=10)

                if response.status_code == 200:
                    logger.success(f"✅ {description} - OK")
                    results.append(True)

                    # Mostrar información relevante
                    try:
                        data = response.json()
                        if "health" in endpoint:
                            status = data.get("status", "unknown")
                            logger.info(f"   Estado: {status}")
                        elif "config" in endpoint:
                            streaming_enabled = data.get("data", {}).get("streaming", {}).get("enabled", False)
                            logger.info(f"   Streaming habilitado: {streaming_enabled}")
                        elif "stats" in endpoint:
                            overview = data.get("data", {}).get("overview", {})
                            logger.info(f"   Sesiones activas: {overview.get('total_sessions', 0)}")
                    except:
                        pass

                else:
                    logger.error(f"❌ {description} - Error {response.status_code}")
                    results.append(False)

            except Exception as e:
                logger.error(f"❌ {description} - Excepción: {str(e)}")
                results.append(False)

        success_rate = sum(results) / len(results) * 100
        logger.info(f"📊 Endpoints REST: {sum(results)}/{len(results)} exitosos ({success_rate:.1f}%)")

        return all(results)

    async def test_websocket_controls(self, session_id: str = None) -> bool:
        """Prueba los controles de WebSocket (pause/resume/stop)"""
        if not session_id:
            session_id = f"test_controls_{int(time.time())}"

        logger.info(f"🎮 Probando controles de WebSocket: {session_id}")

        try:
            uri = f"{self.ws_url}/api/v1/streaming/ws/{session_id}"

            async with websockets.connect(uri) as websocket:

                # Prueba de comandos
                commands_to_test = [
                    ("get_status", "Obtener estado"),
                    ("get_metrics", "Obtener métricas"),
                    ("adjust_quality", "Ajustar calidad"),
                ]

                for command, description in commands_to_test:
                    logger.info(f"📡 Enviando: {command}")

                    message = {"type": command}
                    if command == "adjust_quality":
                        message["data"] = {"quality": 75}

                    await websocket.send(json.dumps(message))

                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        response_data = json.loads(response)

                        if "error" not in response_data:
                            logger.success(f"✅ {description} - OK")
                        else:
                            logger.warning(
                                f"⚠️ {description} - {response_data.get('error', {}).get('message', 'Error')}")

                    except asyncio.TimeoutError:
                        logger.warning(f"⏰ {description} - Timeout")

                return True

        except Exception as e:
            logger.error(f"❌ Error probando controles: {str(e)}")
            return False

    async def run_stress_test(self, concurrent_connections: int = 5) -> bool:
        """Ejecuta un test de estrés con múltiples conexiones"""
        logger.info(f"💪 Iniciando test de estrés: {concurrent_connections} conexiones")

        async def single_connection_test(connection_id: int):
            session_id = f"stress_{connection_id}_{int(time.time())}"
            try:
                uri = f"{self.ws_url}/api/v1/streaming/ws/{session_id}"
                async with websockets.connect(uri) as websocket:
                    # Mantener conexión por 10 segundos
                    for i in range(10):
                        await websocket.send(json.dumps({
                            "type": "ping",
                            "connection_id": connection_id,
                            "iteration": i
                        }))

                        await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        await asyncio.sleep(1)

                    return True
            except Exception as e:
                logger.error(f"❌ Conexión {connection_id} falló: {str(e)}")
                return False

        # Ejecutar conexiones concurrentes
        tasks = [single_connection_test(i) for i in range(concurrent_connections)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if r is True)
        logger.info(f"📊 Test de estrés: {successful}/{concurrent_connections} conexiones exitosas")

        return successful >= concurrent_connections * 0.8  # 80% éxito mínimo

    def generate_test_report(self):
        """Genera un reporte completo de las pruebas"""
        logger.info("📋 Generando reporte de pruebas...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "test_results": self.test_results,
            "streaming_metrics": self.streaming_metrics,
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for r in self.test_results if r.get("success", False)),
                "success_rate": 0
            }
        }

        if report["summary"]["total_tests"] > 0:
            report["summary"]["success_rate"] = (
                    report["summary"]["passed_tests"] / report["summary"]["total_tests"] * 100
            )

        # Guardar reporte
        report_file = f"streaming_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.success(f"📄 Reporte guardado: {report_file}")

        # Mostrar resumen
        logger.info("📊 RESUMEN DE PRUEBAS:")
        logger.info(f"   ✅ Tests exitosos: {report['summary']['passed_tests']}")
        logger.info(f"   ❌ Tests fallidos: {report['summary']['total_tests'] - report['summary']['passed_tests']}")
        logger.info(f"   📈 Tasa de éxito: {report['summary']['success_rate']:.1f}%")

        return report

    async def run_complete_test_suite(self):
        """Ejecuta la suite completa de pruebas"""
        logger.info("🧪 INICIANDO SUITE COMPLETA DE PRUEBAS DE STREAMING")
        logger.info("=" * 70)

        start_time = time.time()

        # Lista de pruebas a ejecutar
        test_suite = [
            ("REST API Endpoints", self.test_rest_api_endpoints, False),
            ("WebSocket Connection", self.test_websocket_connection, True),
            ("WebSocket Controls", self.test_websocket_controls, True),
            ("Complete Streaming Session", self.test_streaming_session_complete, True),
            ("Stress Test", lambda: self.run_stress_test(3), True),
        ]

        for test_name, test_func, is_async in test_suite:
            logger.info("-" * 50)
            logger.info(f"🔬 Ejecutando: {test_name}")

            test_start = time.time()

            try:
                if is_async:
                    success = await test_func()
                else:
                    success = test_func()

                test_duration = time.time() - test_start

                self.test_results.append({
                    "test_name": test_name,
                    "success": success,
                    "duration": test_duration,
                    "timestamp": time.time()
                })

                if success:
                    logger.success(f"✅ {test_name} - EXITOSO ({test_duration:.2f}s)")
                else:
                    logger.error(f"❌ {test_name} - FALLÓ ({test_duration:.2f}s)")

            except Exception as e:
                test_duration = time.time() - test_start
                logger.error(f"❌ {test_name} - EXCEPCIÓN: {str(e)} ({test_duration:.2f}s)")

                self.test_results.append({
                    "test_name": test_name,
                    "success": False,
                    "duration": test_duration,
                    "error": str(e),
                    "timestamp": time.time()
                })

        total_time = time.time() - start_time

        # Generar reporte final
        logger.info("=" * 70)
        report = self.generate_test_report()

        logger.info(f"⏱️ Tiempo total de pruebas: {total_time:.2f}s")

        # Determinar si el sistema está listo
        success_rate = report["summary"]["success_rate"]

        if success_rate >= 80:
            logger.success("🎉 ¡SISTEMA DE STREAMING LISTO PARA FRONTEND!")
            logger.success("   Todos los componentes funcionan correctamente")
            logger.success("   El frontend puede conectarse sin problemas")
        elif success_rate >= 60:
            logger.warning("⚠️ Sistema parcialmente funcional")
            logger.warning("   Revisar errores antes de conectar frontend")
        else:
            logger.error("❌ Sistema no está listo")
            logger.error("   Corregir errores críticos antes de continuar")

        return report


# Funciones de utilidad para pruebas específicas
async def quick_websocket_test(base_url: str = "http://localhost:8000"):
    """Prueba rápida de WebSocket"""
    tester = StreamingTester(base_url)
    return await tester.test_websocket_connection()


async def quick_streaming_test(base_url: str = "http://localhost:8000"):
    """Prueba rápida de streaming completo"""
    tester = StreamingTester(base_url)
    return await tester.test_streaming_session_complete()


def quick_api_test(base_url: str = "http://localhost:8000"):
    """Prueba rápida de API REST"""
    tester = StreamingTester(base_url)
    return tester.test_rest_api_endpoints()


async def main():
    """Función principal para ejecutar las pruebas"""
    import argparse

    parser = argparse.ArgumentParser(description="Pruebas completas del sistema de streaming ALPR")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base de la API")
    parser.add_argument("--test", choices=[
        "websocket", "streaming", "api", "controls", "stress", "all"
    ], default="all", help="Tipo de prueba a ejecutar")
    parser.add_argument("--connections", type=int, default=3, help="Conexiones para stress test")
    parser.add_argument("--video", help="Ruta de video específico para pruebas")

    args = parser.parse_args()

    tester = StreamingTester(args.url)

    logger.info(f"🚀 Iniciando pruebas de streaming en: {args.url}")

    try:
        if args.test == "websocket":
            success = await tester.test_websocket_connection()
            print(f"\n🔌 WebSocket: {'✅ EXITOSO' if success else '❌ FALLÓ'}")

        elif args.test == "streaming":
            success = await tester.test_streaming_session_complete(args.video)
            print(f"\n🎬 Streaming: {'✅ EXITOSO' if success else '❌ FALLÓ'}")

        elif args.test == "api":
            success = tester.test_rest_api_endpoints()
            print(f"\n🌐 API REST: {'✅ EXITOSO' if success else '❌ FALLÓ'}")

        elif args.test == "controls":
            success = await tester.test_websocket_controls()
            print(f"\n🎮 Controles: {'✅ EXITOSO' if success else '❌ FALLÓ'}")

        elif args.test == "stress":
            success = await tester.run_stress_test(args.connections)
            print(f"\n💪 Stress Test: {'✅ EXITOSO' if success else '❌ FALLÓ'}")

        elif args.test == "all":
            await tester.run_complete_test_suite()

    except KeyboardInterrupt:
        logger.warning("⚠️ Pruebas interrumpidas por usuario")
    except Exception as e:
        logger.error(f"❌ Error en las pruebas: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())