#!/usr/bin/env python3
"""
🧪 Cliente de prueba para WebSocket de streaming CARID
Prueba todos los endpoints WebSocket paso a paso
"""

import asyncio
import websockets
import json
import time
import sys


class WebSocketTester:
    def __init__(self, host="localhost", port=8000):
        self.host = host
        self.port = port
        self.base_url = f"ws://{host}:{port}"

    async def test_simple_websocket(self):
        """Prueba el WebSocket ultra-simple"""
        print("🧪 PROBANDO WEBSOCKET SIMPLE")
        print("-" * 40)

        uri = f"{self.base_url}/simple-test"

        try:
            async with websockets.connect(uri) as websocket:
                print(f"✅ Conectado a: {uri}")

                # Recibir mensaje de bienvenida
                welcome = await websocket.recv()
                welcome_data = json.loads(welcome)
                print(f"📥 Bienvenida: {welcome_data['message']}")

                # Enviar ping
                ping_msg = {"type": "ping"}
                await websocket.send(json.dumps(ping_msg))
                print("📤 Enviado: ping")

                # Recibir pong
                pong = await websocket.recv()
                pong_data = json.loads(pong)
                print(f"📥 Respuesta: {pong_data['message']}")

                # Enviar texto simple
                await websocket.send("Hola desde Python!")
                print("📤 Enviado: texto simple")

                # Recibir echo
                echo = await websocket.recv()
                echo_data = json.loads(echo)
                print(f"📥 Echo: {echo_data['echo']}")

                print("✅ Test simple completado exitosamente")
                return True

        except Exception as e:
            print(f"❌ Error en test simple: {str(e)}")
            return False

    async def test_streaming_test_endpoint(self):
        """Prueba el endpoint de test de streaming"""
        print("\n🎬 PROBANDO WEBSOCKET DE STREAMING TEST")
        print("-" * 45)

        session_id = f"test_{int(time.time())}"
        uri = f"{self.base_url}/api/v1/streaming/test/{session_id}"

        try:
            async with websockets.connect(uri) as websocket:
                print(f"✅ Conectado a: {uri}")

                # Recibir confirmación
                welcome = await websocket.recv()
                welcome_data = json.loads(welcome)
                print(f"📥 Confirmación: {welcome_data['message']}")

                # Enviar mensaje de prueba
                test_msg = {
                    "type": "test",
                    "message": "Probando streaming",
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(test_msg))
                print("📤 Enviado: mensaje de prueba")

                # Recibir echo
                echo = await websocket.recv()
                echo_data = json.loads(echo)
                print(f"📥 Echo recibido: {echo_data['type']}")

                print("✅ Test de streaming completado exitosamente")
                return True

        except Exception as e:
            print(f"❌ Error en test de streaming: {str(e)}")
            return False

    async def test_main_streaming_endpoint(self):
        """Prueba el endpoint principal de streaming"""
        print("\n🎯 PROBANDO WEBSOCKET PRINCIPAL DE STREAMING")
        print("-" * 50)

        session_id = f"main_{int(time.time())}"
        uri = f"{self.base_url}/api/v1/streaming/ws/{session_id}"

        try:
            async with websockets.connect(uri) as websocket:
                print(f"✅ Conectado a: {uri}")

                # Recibir configuración inicial
                config = await websocket.recv()
                config_data = json.loads(config)
                print(f"📥 Configuración: {config_data['type']}")
                print(f"   📊 Servicio: {config_data.get('server_info', {}).get('service', 'N/A')}")

                # Solicitar estado
                status_msg = {"type": "get_status"}
                await websocket.send(json.dumps(status_msg))
                print("📤 Solicitado: estado de sesión")

                # Recibir estado
                status = await websocket.recv()
                status_data = json.loads(status)
                print(f"📥 Estado: {status_data['data']['status']}")
                print(f"   🆔 Session ID: {status_data['data']['session_id']}")

                # Enviar ping
                ping_msg = {"type": "ping"}
                await websocket.send(json.dumps(ping_msg))
                print("📤 Enviado: ping")

                # Recibir pong
                pong = await websocket.recv()
                pong_data = json.loads(pong)
                print(f"📥 Pong recibido: {pong_data['type']}")

                print("✅ Test principal completado exitosamente")
                return True

        except Exception as e:
            print(f"❌ Error en test principal: {str(e)}")
            return False

    async def test_all_endpoints(self):
        """Ejecuta todos los tests"""
        print("🚀 INICIANDO TESTS DE WEBSOCKET CARID")
        print("=" * 60)

        results = []

        # Test 1: WebSocket simple
        result1 = await self.test_simple_websocket()
        results.append(("Simple WebSocket", result1))

        # Test 2: Streaming test
        result2 = await self.test_streaming_test_endpoint()
        results.append(("Streaming Test", result2))

        # Test 3: Streaming principal
        result3 = await self.test_main_streaming_endpoint()
        results.append(("Streaming Principal", result3))

        # Resumen
        print("\n📊 RESUMEN DE TESTS")
        print("=" * 30)

        passed = 0
        for test_name, passed_test in results:
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"{status} {test_name}")
            if passed_test:
                passed += 1

        print(f"\n🎯 Resultado: {passed}/{len(results)} tests pasaron")

        if passed == len(results):
            print("🎉 ¡Todos los tests pasaron! WebSocket funcionando correctamente")
            return True
        else:
            print("⚠️ Algunos tests fallaron. Revisar configuración del servidor")
            return False


async def main():
    """Función principal"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("Uso: python test_websocket_client.py [host] [port]")
            print("Ejemplo: python test_websocket_client.py localhost 8000")
            return

        host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    else:
        host = "localhost"
        port = 8000

    print(f"🌐 Probando servidor: {host}:{port}")
    print(f"📅 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    tester = WebSocketTester(host, port)

    try:
        success = await tester.test_all_endpoints()

        if success:
            print("\n🎊 ÉXITO: Sistema WebSocket funcionando perfectamente")
            print("🔥 ¡Listo para streaming en tiempo real!")
        else:
            print("\n💔 FALLO: Algunos endpoints no funcionan")
            print("🔧 Revisar logs del servidor para más detalles")

    except KeyboardInterrupt:
        print("\n⏹️ Tests interrumpidos por el usuario")
    except Exception as e:
        print(f"\n💥 Error inesperado: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())