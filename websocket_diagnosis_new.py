#!/usr/bin/env python3
"""
🔍 Diagnóstico WebSocket CARID - Versión para nuevo sistema
"""

import requests
import json
import asyncio
import websockets
import sys
import time


class CARIDWebSocketDiagnostic:
    def __init__(self, host="localhost", port=8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_base_url = f"ws://{host}:{port}"

    def test_server_basic(self):
        """Test básico del servidor"""
        print("🔍 1. VERIFICANDO SERVIDOR BÁSICO...")

        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            print(f"   ✅ Servidor responde: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"   📊 Servicio: {data.get('service', 'N/A')}")
                print(f"   🔢 Versión: {data.get('version', 'N/A')}")
                print(f"   🤖 Modelos: {'✅' if data.get('models_loaded') else '❌'}")
                print(f"   🎬 Streaming: {'✅' if data.get('streaming_enabled') else '❌'}")
                return True

            return False

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return False

    def test_rest_endpoints(self):
        """Test endpoints REST"""
        print("\n📡 2. VERIFICANDO ENDPOINTS REST...")

        endpoints = [
            ("/api/v1/health", "Health Check"),
            ("/api/v1/streaming/health", "Streaming Health"),
            ("/api/v1/streaming/sessions", "Sessions List"),
            ("/api/v1/streaming/test-connection", "Test Connection"),
            ("/status", "Quick Status")
        ]

        success_count = 0

        for endpoint, name in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print(f"   ✅ {name}: OK")
                    success_count += 1
                else:
                    print(f"   ❌ {name}: {response.status_code}")
            except Exception as e:
                print(f"   ❌ {name}: Error - {str(e)}")

        print(f"   📊 Resultado: {success_count}/{len(endpoints)} endpoints funcionando")
        return success_count == len(endpoints)

    async def test_websocket_endpoints(self):
        """Test endpoints WebSocket"""
        print("\n🔌 3. VERIFICANDO ENDPOINTS WEBSOCKET...")

        # Test 1: WebSocket simple
        print("   🧪 Probando WebSocket simple...")
        simple_success = await self._test_simple_websocket()

        # Test 2: Streaming test
        print("   🎬 Probando streaming test...")
        test_success = await self._test_streaming_test()

        # Test 3: Streaming principal
        print("   🎯 Probando streaming principal...")
        main_success = await self._test_streaming_main()

        total_tests = 3
        passed_tests = sum([simple_success, test_success, main_success])

        print(f"   📊 Resultado WebSocket: {passed_tests}/{total_tests} funcionando")

        return passed_tests >= 2  # Al menos 2 de 3 deben funcionar

    async def _test_simple_websocket(self):
        """Test WebSocket simple"""
        try:
            uri = f"{self.ws_base_url}/simple-test"

            async with websockets.connect(uri, timeout=5) as websocket:
                # Recibir bienvenida
                welcome = await asyncio.wait_for(websocket.recv(), timeout=3)
                welcome_data = json.loads(welcome)

                if welcome_data.get("type") == "connected":
                    print("      ✅ Simple WebSocket OK")
                    return True

        except Exception as e:
            print(f"      ❌ Simple WebSocket: {str(e)}")

        return False

    async def _test_streaming_test(self):
        """Test streaming test endpoint"""
        try:
            session_id = f"diag_{int(time.time())}"
            uri = f"{self.ws_base_url}/api/v1/streaming/test/{session_id}"

            async with websockets.connect(uri, timeout=5) as websocket:
                # Recibir confirmación
                confirmation = await asyncio.wait_for(websocket.recv(), timeout=3)
                data = json.loads(confirmation)

                if data.get("type") == "test_connected":
                    print("      ✅ Streaming Test OK")
                    return True

        except Exception as e:
            print(f"      ❌ Streaming Test: {str(e)}")

        return False

    async def _test_streaming_main(self):
        """Test streaming principal"""
        try:
            session_id = f"main_diag_{int(time.time())}"
            uri = f"{self.ws_base_url}/api/v1/streaming/ws/{session_id}"

            async with websockets.connect(uri, timeout=5) as websocket:
                # Recibir configuración
                config = await asyncio.wait_for(websocket.recv(), timeout=3)
                data = json.loads(config)

                if data.get("type") == "connection_established":
                    print("      ✅ Streaming Principal OK")
                    return True

        except Exception as e:
            print(f"      ❌ Streaming Principal: {str(e)}")

        return False

    def test_dependencies(self):
        """Test dependencias"""
        print("\n📦 4. VERIFICANDO DEPENDENCIAS...")

        deps_ok = True

        try:
            import websockets
            print(f"   ✅ websockets: {websockets.__version__}")
        except ImportError:
            print("   ❌ websockets: NO INSTALADO")
            deps_ok = False

        try:
            import uvicorn
            print(f"   ✅ uvicorn: {uvicorn.__version__}")
        except ImportError:
            print("   ❌ uvicorn: NO INSTALADO")
            deps_ok = False

        try:
            import fastapi
            print(f"   ✅ fastapi: {fastapi.__version__}")
        except ImportError:
            print("   ❌ fastapi: NO INSTALADO")
            deps_ok = False

        return deps_ok

    def get_streaming_info(self):
        """Obtiene información de streaming"""
        print("\n🎬 5. INFORMACIÓN DE STREAMING...")

        try:
            response = requests.get(f"{self.base_url}/api/v1/streaming/health", timeout=5)

            if response.status_code == 200:
                data = response.json()
                print(f"   📊 Estado: {data.get('status', 'unknown')}")

                sessions = data.get('sessions', {})
                print(f"   🔌 Sesiones activas: {sessions.get('active', 0)}")
                print(f"   📈 Capacidad: {sessions.get('capacity_usage', 0):.1f}%")

                models = data.get('models', {})
                print(f"   🤖 Modelos cargados: {'✅' if models.get('loaded') else '❌'}")

                caps = data.get('capabilities', {})
                print(f"   🎯 WebSocket streaming: {'✅' if caps.get('websocket_streaming') else '❌'}")
                print(f"   ⚡ Procesamiento tiempo real: {'✅' if caps.get('real_time_processing') else '❌'}")

                return True
            else:
                print(f"   ❌ Error obteniendo info: {response.status_code}")
                return False

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return False

    async def run_full_diagnosis(self):
        """Ejecuta diagnóstico completo"""
        print("🔍 DIAGNÓSTICO WEBSOCKET CARID")
        print("=" * 50)

        results = {}

        # Tests secuenciales
        results['server_basic'] = self.test_server_basic()
        results['rest_endpoints'] = self.test_rest_endpoints()
        results['websocket_endpoints'] = await self.test_websocket_endpoints()
        results['dependencies'] = self.test_dependencies()
        results['streaming_info'] = self.get_streaming_info()

        # Resumen
        print("\n📊 RESUMEN DE DIAGNÓSTICO")
        print("=" * 30)

        passed = 0
        total = len(results)

        for test_name, result in results.items():
            status = "✅" if result else "❌"
            formatted_name = test_name.replace('_', ' ').title()
            print(f"{status} {formatted_name}")
            if result:
                passed += 1

        print(f"\n🎯 Resultado final: {passed}/{total} tests pasaron")

        # Diagnóstico y recomendaciones
        self.provide_recommendations(results)

        return passed == total

    def provide_recommendations(self, results):
        """Proporciona recomendaciones basadas en resultados"""
        print("\n🔧 RECOMENDACIONES:")
        print("-" * 25)

        if not results.get('server_basic'):
            print("❌ SERVIDOR NO RESPONDE")
            print("   💡 Verificar que el servidor esté ejecutándose")
            print("   💡 Comando: python main.py")
            print()

        if not results.get('dependencies'):
            print("❌ DEPENDENCIAS FALTANTES")
            print("   💡 Instalar: pip install websockets uvicorn[standard] fastapi")
            print()

        if not results.get('websocket_endpoints'):
            print("❌ ENDPOINTS WEBSOCKET NO FUNCIONAN")
            print("   💡 Verificar que uvicorn se ejecute con ws='auto'")
            print("   💡 Revisar que streaming_router esté incluido")
            print("   💡 Verificar logs del servidor para errores")
            print()

        if all(results.values()):
            print("🎉 ¡TODO FUNCIONANDO PERFECTAMENTE!")
            print("   🚀 Sistema listo para streaming en tiempo real")
            print("   🔗 Conecta a: ws://localhost:8000/api/v1/streaming/ws/{session_id}")
            print()

        print("📚 DOCUMENTACIÓN:")
        print("   🌐 API Docs: http://localhost:8000/docs")
        print("   📊 Estado: http://localhost:8000/status")
        print("   🏥 Health: http://localhost:8000/api/v1/streaming/health")


async def main():
    """Función principal"""
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("Uso: python websocket_diagnosis_new.py [host] [port]")
        print("Ejemplo: python websocket_diagnosis_new.py localhost 8000")
        return

    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    print(f"🌐 Diagnosticando: {host}:{port}")
    print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    diagnostic = CARIDWebSocketDiagnostic(host, port)

    try:
        success = await diagnostic.run_full_diagnosis()

        if success:
            print(f"\n✅ DIAGNÓSTICO COMPLETADO - TODO OK")
            sys.exit(0)
        else:
            print(f"\n⚠️ DIAGNÓSTICO COMPLETADO - PROBLEMAS ENCONTRADOS")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⏹️ Diagnóstico interrumpido")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())