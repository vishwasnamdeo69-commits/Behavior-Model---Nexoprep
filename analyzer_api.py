import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ANALYZER_API_HOST = "127.0.0.1"
ANALYZER_API_PORT = 8765


class AnalyzerAPIHandler(BaseHTTPRequestHandler):
    controller = None

    def log_message(self, format, *args):
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _idle_status(self) -> dict:
        return {
            "running": False,
            "bootReady": False,
            "sessionActive": False,
            "elapsed": 0,
            "shutdownRequested": False,
            "sessionState": "idle",
            "lifecycleId": None,
        }

    def do_GET(self) -> None:
        if self.path != "/status":
            self._send_json(404, {"error": "not found"})
            return

        controller = self.controller
        if controller is None:
            self._send_json(200, self._idle_status())
            return

        self._send_json(200, controller.get_status())

    def do_POST(self) -> None:
        controller = self.controller
        if controller is None:
            self._send_json(503, {"ok": False, "error": "controller unavailable"})
            return

        if self.path == "/start":
            print("[ANALYZER_API] start requested")
            result = controller.start()
            self._send_json(200, result)
            return

        if self.path == "/stop":
            print("[ANALYZER_API] stop requested")
            result = controller.stop()
            status = controller.get_status()
            self._send_json(200, {
                **result,
                "bootReady": status["bootReady"],
                "sessionState": status["sessionState"],
            })
            return

        if self.path == "/quit":
            print("[ANALYZER_API] quit requested")
            controller.quit()
            status = controller.get_status()
            self._send_json(200, {
                "ok": True,
                "shutdownRequested": status["shutdownRequested"],
                "sessionState": status["sessionState"],
                "bootReady": status["bootReady"],
            })
            return

        self._send_json(404, {"error": "not found"})


def start_analyzer_api_server(controller) -> HTTPServer:
    AnalyzerAPIHandler.controller = controller
    server = HTTPServer((ANALYZER_API_HOST, ANALYZER_API_PORT), AnalyzerAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="AnalyzerAPIServer")
    thread.start()
    print("[ANALYZER_API] server started")
    return server
