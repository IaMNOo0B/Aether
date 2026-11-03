"""Read-only localhost operational dashboard using the Python standard library."""
from __future__ import annotations

from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any
from urllib.parse import urlsplit

from aether.resources import ResourceMonitor
from aether.storage import Store


class AetherDashboard:
    def __init__(
        self,
        store: Store,
        *,
        model_provider: str = "unknown",
        model_id: str = "unknown",
        host: str = "127.0.0.1",
        port: int = 8765,
        monitor: ResourceMonitor | None = None,
    ):
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Dashboard must bind to loopback only")
        if not 0 <= port <= 65535:
            raise ValueError("Dashboard port must be between 0 and 65535")
        self.store = store
        self.model_provider = model_provider
        self.model_id = model_id
        self.host = host
        self.port = port
        self.monitor = monitor or ResourceMonitor()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        tasks = self.store.list_tasks(limit=50)
        notifications = self.store.list_notifications(unread_only=True, limit=50)
        return {
            "system": "AETHER",
            "status": "operational",
            "model": {"provider": self.model_provider, "id": self.model_id},
            "resources": self.monitor.snapshot(cpu_sample_seconds=0.05),
            "tasks": [
                {
                    "id": item["id"],
                    "status": item["status"],
                    "objective": str(item.get("objective", ""))[:160],
                    "updated_at": item.get("updated_at"),
                }
                for item in tasks
            ],
            "notifications": [
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    "title": item["title"],
                    "severity": item["severity"],
                    "created_at": item["created_at"],
                }
                for item in notifications
            ],
            "safety": {
                "emergency_stop": self.store.emergency_stop_enabled(),
                "mutating_api": False,
                "binding": "loopback-only",
            },
        }

    @staticmethod
    def _page() -> bytes:
        page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'">
<title>AETHER — Operational Dashboard</title>
<style>
body{font:15px system-ui,sans-serif;background:#10151d;color:#e8edf4;margin:2rem;max-width:1100px}
h1{letter-spacing:.08em}section{background:#192330;border:1px solid #344252;border-radius:10px;padding:1rem;margin:1rem 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#c8d5e4}.muted{color:#9aaabc}
</style></head><body>
<h1>AETHER <span class="muted">OPERATIONAL STATUS</span></h1>
<p class="muted">Read-only local dashboard. Refreshes every five seconds.</p>
<section><h2>Runtime</h2><pre id="runtime">Loading…</pre></section>
<section><h2>Resources</h2><pre id="resources">Loading…</pre></section>
<section><h2>Tasks</h2><pre id="tasks">Loading…</pre></section>
<section><h2>Unread notifications</h2><pre id="notifications">Loading…</pre></section>
<script>
async function refresh(){try{
const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw new Error('HTTP '+r.status);
const d=await r.json();
document.getElementById('runtime').textContent=JSON.stringify({status:d.status,model:d.model,safety:d.safety},null,2);
document.getElementById('resources').textContent=JSON.stringify(d.resources,null,2);
document.getElementById('tasks').textContent=JSON.stringify(d.tasks,null,2);
document.getElementById('notifications').textContent=JSON.stringify(d.notifications,null,2);
}catch(e){document.getElementById('runtime').textContent='Dashboard unavailable: '+e.message;}}
refresh();setInterval(refresh,5000);
</script></body></html>"""
        return page.encode("utf-8")

    def start(self) -> tuple[str, int]:
        if self._server is not None:
            raise RuntimeError("Dashboard is already running")
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/":
                    payload = dashboard._page()
                    content_type = "text/html; charset=utf-8"
                    status = 200
                elif path == "/api/status":
                    payload = json.dumps(
                        dashboard.snapshot(), ensure_ascii=False, default=str
                    ).encode("utf-8")
                    content_type = "application/json; charset=utf-8"
                    status = 200
                else:
                    payload = b"Not found"
                    content_type = "text/plain; charset=utf-8"
                    status = 404
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                    "connect-src 'self'; object-src 'none'; base-uri 'none'",
                )
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:
                self.send_error(405, "Read-only dashboard")

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="aether-dashboard", daemon=True
        )
        self._thread.start()
        return self.host, self.port

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None
