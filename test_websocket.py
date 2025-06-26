#!/usr/bin/env python3
"""
Script para probar el WebSocket de streaming ALPR sin frontend
Permite conectar, subir video y recibir resultados en tiempo real
"""

import asyncio
import websockets
import requests
import json
import time
import os
import sys
from pathlib import Path
from typing import Optional


class ALPRWebSocketTester:
    def __init__(self,
                 server_host: str = "localhost",
                 server_port: int = 8000,
                 session_id: Optional[str] = None):
        self.server_host = server_host
        self.server_port = server_port
        self.session_id = session_id or f"test_session_{int(time.time())}"

        # URLs
        self.ws_url = f"ws://{server_host}:{server_port}/api/v1/streaming/ws/{self.session_id}"
        self.http_base = f"http://{server_host}:{server_port}/api/v1"

        # Control
        self.websocket = None
        self.connected = False
        self.processing_stats = {
            "frames_received": 0,
            "detections_received": 0,
            "unique_plates": set(),
            "start_time": None,
            "last_update": None
        }

    async def connect_websocket(self):
        """Conecta al WebSocket del servidor"""
        try:
            print(f"🔌 Conectando a WebSocket: {self.ws_url}")

            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )

            self.connected = True
            print(f"✅ WebSocket conectado! Session ID: {self.session_id}")

            # Esperar mensaje de configuración inicial
            initial_message = await self.websocket.recv()
            config_data = json.loads(initial_message)

            if config_data.get("type") == "streaming_config":
                print("📋 Configuración del servidor recibida:")
                print(f"   - Max conexiones: {config_data['data'].get('websocket', {}).get('max_connections', 'N/A')}")
                print(f"   - Calidad de frame: {config_data['data'].get('frame_processing', {}).get('quality', 'N/A')}")
                print(f"   - Streaming habilitado: {config_data['data'].get('enabled', 'N/A')}")

            return True

        except Exception as e:
            print(f"❌ Error conectando WebSocket: {str(e)}")
            self.connected = False
            return False

    async def upload_video(self, video_path: str, **params):
        """Sube video para procesamiento vía HTTP POST"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video no encontrado: {video_path}")

            print(f"📹 Subiendo video: {video_path}")
            print(f"📊 Parámetros: {params}")

            # Preparar archivos y datos
            with open(video_path, 'rb') as video_file:
                files = {'file': (os.path.basename(video_path), video_file, 'video/mp4')}
                data = {
                    'session_id': self.session_id,
                    'confidence_threshold': params.get('confidence_threshold', 0.3),
                    'iou_threshold': params.get('iou_threshold', 0.4),
                    'frame_skip': params.get('frame_skip', 2),
                    'max_duration': params.get('max_duration', 600),
                    'send_all_frames': params.get('send_all_frames', False),
                    'adaptive_quality': params.get('adaptive_quality', True),
                    'enable_thumbnails': params.get('enable_thumbnails', True)
                }

                url = f"{self.http_base}/streaming/start-session"
                response = requests.post(url, files=files, data=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                print("✅ Video subido exitosamente!")
                print(f"   - Archivo: {result['data']['file_info']['filename']}")
                print(f"   - Tamaño: {result['data']['file_info']['size_mb']}MB")
                print("📡 Procesamiento iniciado, escuchando WebSocket...")
                return True
            else:
                print(f"❌ Error subiendo video: {response.status_code}")
                print(f"   Respuesta: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Error en upload_video: {str(e)}")
            return False

    async def listen_websocket(self):
        """Escucha mensajes del WebSocket en tiempo real"""
        try:
            self.processing_stats["start_time"] = time.time()

            while self.connected and self.websocket:
                try:
                    # Recibir mensaje con timeout
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                    data = json.loads(message)

                    await self.handle_websocket_message(data)

                except asyncio.TimeoutError:
                    # Enviar ping para mantener conexión
                    if self.websocket:
                        await self.websocket.send(json.dumps({
                            "type": "ping",
                            "timestamp": time.time()
                        }))
                    continue

                except websockets.exceptions.ConnectionClosed:
                    print("🔌 Conexión WebSocket cerrada")
                    break

        except Exception as e:
            print(f"❌ Error escuchando WebSocket: {str(e)}")
        finally:
            self.connected = False

    async def handle_websocket_message(self, data: dict):
        """Maneja diferentes tipos de mensajes del WebSocket"""
        message_type = data.get("type", "unknown")

        if message_type == "streaming_started":
            print("🚀 Procesamiento iniciado!")
            video_info = data.get("data", {}).get("video_info", {})
            print(f"   - Duración: {video_info.get('duration_seconds', 0):.1f}s")
            print(f"   - Frames totales: {video_info.get('frame_count', 0)}")
            print(f"   - FPS: {video_info.get('fps', 0):.1f}")

        elif message_type == "streaming_update":
            await self.handle_streaming_update(data.get("data", {}))

        elif message_type == "streaming_completed":
            await self.handle_completion(data.get("data", {}))

        elif message_type == "streaming_error":
            print(f"❌ Error en streaming: {data.get('error', 'Error desconocido')}")

        elif message_type == "pong":
            # Respuesta a ping, no hacer nada
            pass

        elif message_type == "ping":
            # Responder al ping del servidor
            await self.websocket.send(json.dumps({
                "type": "pong",
                "timestamp": time.time()
            }))

        else:
            print(f"📥 Mensaje recibido ({message_type}): {json.dumps(data, indent=2)}")

    async def handle_streaming_update(self, data: dict):
        """Maneja actualizaciones de progreso del streaming"""
        frame_info = data.get("frame_info", {})
        progress = data.get("progress", {})
        current_detections = data.get("current_detections", [])
        detection_summary = data.get("detection_summary", {})

        # Actualizar estadísticas
        self.processing_stats["frames_received"] += 1
        self.processing_stats["last_update"] = time.time()

        if current_detections:
            self.processing_stats["detections_received"] += len(current_detections)
            for detection in current_detections:
                self.processing_stats["unique_plates"].add(detection.get("plate_text", ""))

        # Mostrar progreso cada 10 frames o si hay detecciones
        should_show = (
                self.processing_stats["frames_received"] % 10 == 0 or
                len(current_detections) > 0
        )

        if should_show:
            elapsed = time.time() - self.processing_stats["start_time"]

            print(f"\n📊 PROGRESO - Frame {frame_info.get('frame_number', 0)}")
            print(f"   ⏱️  Tiempo: {elapsed:.1f}s")
            print(f"   📈 Progreso: {progress.get('progress_percent', 0):.1f}%")
            print(f"   ⚡ Velocidad: {progress.get('processing_speed', 0):.2f} fps")
            print(f"   🎯 Detecciones totales: {detection_summary.get('total_detections', 0)}")
            print(f"   🚗 Placas únicas: {detection_summary.get('unique_plates_count', 0)}")

            if current_detections:
                print(f"   🆕 DETECCIONES EN ESTE FRAME:")
                for det in current_detections:
                    plate_text = det.get("plate_text", "N/A")
                    confidence = det.get("overall_confidence", 0)
                    is_valid = "✅" if det.get("is_valid_plate", False) else "⚠️"
                    print(f"      {is_valid} {plate_text} (Conf: {confidence:.3f})")

    async def handle_completion(self, data: dict):
        """Maneja la finalización del procesamiento"""
        print("\n🎉 ¡PROCESAMIENTO COMPLETADO!")

        total_time = data.get("total_processing_time", 0)
        detection_summary = data.get("detection_summary", {})

        print(f"⏱️  Tiempo total: {total_time:.1f}s")
        print(f"📹 Frames procesados: {data.get('frames_processed', 0)}")
        print(f"🎯 Frames con detecciones: {data.get('frames_with_detections', 0)}")
        print(f"🚗 Placas únicas encontradas: {detection_summary.get('unique_plates_count', 0)}")

        # Mostrar mejores placas
        best_plates = detection_summary.get('best_plates', [])
        if best_plates:
            print(f"\n🏆 TOP PLACAS DETECTADAS:")
            for i, plate in enumerate(best_plates[:5], 1):
                is_valid = "✅" if plate.get("is_valid_format", False) else "⚠️"
                print(f"   {i}. {is_valid} {plate.get('plate_text', 'N/A')} "
                      f"(Conf: {plate.get('best_confidence', 0):.3f}, "
                      f"Detecciones: {plate.get('detection_count', 0)})")

        # Estadísticas finales del cliente
        client_elapsed = time.time() - self.processing_stats["start_time"]
        print(f"\n📊 ESTADÍSTICAS DEL CLIENTE:")
        print(f"   📡 Updates recibidos: {self.processing_stats['frames_received']}")
        print(f"   🎯 Detecciones recibidas: {self.processing_stats['detections_received']}")
        print(f"   🚗 Placas únicas vistas: {len(self.processing_stats['unique_plates'])}")
        print(f"   ⏱️  Tiempo total: {client_elapsed:.1f}s")

    async def send_control_command(self, command: str):
        """Envía comandos de control (pause, resume, stop)"""
        if not self.connected or not self.websocket:
            print("❌ WebSocket no conectado")
            return False

        try:
            message = {
                "type": f"{command}_processing",
                "timestamp": time.time()
            }
            await self.websocket.send(json.dumps(message))
            print(f"📤 Comando enviado: {command}")
            return True
        except Exception as e:
            print(f"❌ Error enviando comando: {str(e)}")
            return False

    async def get_status(self):
        """Solicita el estado actual del procesamiento"""
        if not self.connected or not self.websocket:
            print("❌ WebSocket no conectado")
            return False

        try:
            message = {
                "type": "get_status",
                "timestamp": time.time()
            }
            await self.websocket.send(json.dumps(message))
            print("📤 Solicitando estado...")
            return True
        except Exception as e:
            print(f"❌ Error solicitando estado: {str(e)}")
            return False

    async def disconnect(self):
        """Desconecta el WebSocket"""
        if self.websocket:
            try:
                await self.websocket.close()
                print("🔌 WebSocket desconectado")
            except Exception as e:
                print(f"⚠️ Error desconectando: {str(e)}")

        self.connected = False
        self.websocket = None

    def get_server_info(self):
        """Obtiene información del servidor vía HTTP"""
        try:
            response = requests.get(f"{self.http_base}/streaming/health", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Error obteniendo info del servidor: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error conectando al servidor: {str(e)}")
            return None


async def interactive_mode(tester: ALPRWebSocketTester):
    """Modo interactivo para controlar el procesamiento"""
    print("\n🎮 MODO INTERACTIVO ACTIVADO")
    print("Comandos disponibles:")
    print("  'pause' - Pausar procesamiento")
    print("  'resume' - Reanudar procesamiento")
    print("  'stop' - Detener procesamiento")
    print("  'status' - Ver estado actual")
    print("  'quit' - Salir")
    print("\nEscribe un comando y presiona Enter:")

    while tester.connected:
        try:
            # Leer input del usuario (no bloqueante)
            command = input("> ").strip().lower()

            if command == 'quit':
                break
            elif command in ['pause', 'resume', 'stop']:
                await tester.send_control_command(command)
            elif command == 'status':
                await tester.get_status()
            elif command:
                print(f"❌ Comando desconocido: {command}")

        except KeyboardInterrupt:
            break
        except EOFError:
            break


async def main():
    """Función principal del script"""
    print("🚗 ALPR WebSocket Tester")
    print("=" * 50)

    # Configuración
    if len(sys.argv) < 2:
        print("❌ Uso: python script.py <ruta_del_video> [opciones]")
        print("\nEjemplo:")
        print("  python script.py video.mp4")
        print("  python script.py video.mp4 --host localhost --port 8000")
        print("  python script.py video.mp4 --confidence 0.4 --frame-skip 3")
        return

    video_path = sys.argv[1]

    # Parámetros opcionales
    server_host = "localhost"
    server_port = 8000

    processing_params = {
        'confidence_threshold': 0.3,
        'iou_threshold': 0.4,
        'frame_skip': 2,
        'max_duration': 600,
        'send_all_frames': False,
        'adaptive_quality': True,
        'enable_thumbnails': True
    }

    # Procesar argumentos
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--host' and i + 1 < len(sys.argv):
            server_host = sys.argv[i + 1]
            i += 2
        elif arg == '--port' and i + 1 < len(sys.argv):
            server_port = int(sys.argv[i + 1])
            i += 2
        elif arg == '--confidence' and i + 1 < len(sys.argv):
            processing_params['confidence_threshold'] = float(sys.argv[i + 1])
            i += 2
        elif arg == '--frame-skip' and i + 1 < len(sys.argv):
            processing_params['frame_skip'] = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    # Crear tester
    tester = ALPRWebSocketTester(server_host, server_port)

    try:
        # Verificar servidor
        print("🔍 Verificando servidor...")
        server_info = tester.get_server_info()
        if server_info:
            print("✅ Servidor ALPR disponible")
            print(f"   - Estado: {server_info.get('status', 'unknown')}")
            print(f"   - Versión: {server_info.get('version', 'unknown')}")
            print(f"   - Streaming habilitado: {server_info.get('configuration', {}).get('streaming_enabled', False)}")
        else:
            print("❌ Servidor no disponible. ¿Está ejecutándose?")
            return

        # Conectar WebSocket
        if not await tester.connect_websocket():
            return

        # Crear tarea para escuchar WebSocket
        listen_task = asyncio.create_task(tester.listen_websocket())

        # Esperar un momento para establecer conexión
        await asyncio.sleep(1)

        # Subir video
        if await tester.upload_video(video_path, **processing_params):
            print("\n🎮 Presiona Ctrl+C en cualquier momento para interactuar")

            # Crear tarea para modo interactivo
            try:
                # Esperar a que termine el procesamiento o input del usuario
                await listen_task
            except KeyboardInterrupt:
                print("\n🎮 Modo interactivo activado...")
                # Cancelar listening y empezar modo interactivo
                listen_task.cancel()

                # Continuar escuchando en background
                new_listen_task = asyncio.create_task(tester.listen_websocket())

                # Modo interactivo
                await interactive_mode(tester)

                # Cancelar tarea de listening
                new_listen_task.cancel()

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        await tester.disconnect()
        print("👋 ¡Hasta luego!")


if __name__ == "__main__":
    asyncio.run(main())