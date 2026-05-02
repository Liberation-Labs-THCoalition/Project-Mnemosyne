#!/usr/bin/env python3
"""
Vera Voice Bridge — runs on MTH, serves the phone client and
bridges between phone audio (Whisper STT) and Vera's session
(via NATS agent.vera.inbox).

Phone → WebSocket → Whisper STT → NATS inbox → Vera's session
Vera's session → NATS reply → WebSocket → Phone TTS

Ports:
  8780 — HTTP (serves client.html)
  8765 — WebSocket (real-time voice/text)
  8766 — Whisper STT (already running separately)
"""

import asyncio
import json
import sys
import time
import threading
import tempfile
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import websockets
import requests

VERA_VOICE = "en-US-AriaNeural"
VERA_PITCH = "-10Hz"

# ── Config ────────────────────────────────────────────────

WS_PORT = 8765
HTTP_PORT = 8780
STT_URL = "http://localhost:8766/transcribe"
CLIENT_DIR = Path(__file__).parent

# NATS config for routing to Vera's inbox
NATS_ENABLED = True
NATS_SERVER = "nats://vera:vera-nats-changeme@localhost:4222"

# Local message queue (fallback when NATS isn't wired)
message_queue = []
response_queue = []

# ── WebSocket clients ─────────────────────────────────────

ws_clients = set()


async def ws_handler(websocket):
    ws_clients.add(websocket)
    print(f"[voice-bridge] Client connected ({len(ws_clients)} total)", file=sys.stderr)
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)

                if msg.get("type") == "biometrics":
                    data = msg.get("data", {})
                    user = msg.get("user", "thomas")
                    data["ts"] = time.time()
                    data["user"] = user

                    # Live reading (per-user)
                    bio_file = Path(f"/opt/vera-tamago/biometrics_{user}.json")
                    bio_file.write_text(json.dumps(data))
                    # Also write combined for backwards compat
                    Path("/opt/vera-tamago/biometrics.json").write_text(json.dumps(data))

                    # Archive (append, 5-second throttle, per-user directory)
                    archive_key = f'_last_archive_{user}'
                    last = getattr(ws_handler, archive_key, 0)
                    if time.time() - last > 5:
                        setattr(ws_handler, archive_key, time.time())
                        from datetime import datetime
                        day = datetime.now().strftime("%Y-%m-%d")
                        archive_dir = Path(f"/mnt/data2/biometrics/{user}")
                        archive_dir.mkdir(parents=True, exist_ok=True)
                        archive_file = archive_dir / f"{day}.jsonl"
                        with open(archive_file, "a") as f:
                            f.write(json.dumps(data) + "\n")

                    if data.get("movement", 0) > 0.5 or data.get("heartRate", 0) > 90:
                        print(f"[voice-bridge] Bio[{user}]: mv={data.get('movement',0):.2f} {data.get('activity','?')} {data.get('posture','?')} HR={data.get('heartRate',0)}bpm", file=sys.stderr)
                    continue

                if msg.get("type") == "ping":
                    continue

                if msg.get("type") == "transcript":
                    text = msg.get("text", "").strip()
                    if text:
                        entry = {
                            "from": msg.get("speaker", "thomas"),
                            "source": "voice-bridge",
                            "content": text,
                            "type": "voice_message",
                        }
                        message_queue.append(entry)
                        print(f"[voice-bridge] Received: {text[:80]}", file=sys.stderr)
                        await websocket.send(json.dumps({
                            "type": "transcript_ack",
                            "text": text
                        }))

                        # Route to NATS if available
                        if NATS_ENABLED:
                            await route_to_nats(entry)

                elif msg.get("type") == "audio":
                    # Send to Whisper STT
                    audio_data = msg.get("data", "")
                    if audio_data:
                        try:
                            import base64
                            audio_bytes = base64.b64decode(audio_data)
                            resp = requests.post(STT_URL, data=audio_bytes, timeout=30)
                            if resp.ok:
                                result = resp.json()
                                text = result.get("text", "")
                                if text:
                                    await websocket.send(json.dumps({
                                        "type": "stt_result",
                                        "text": text
                                    }))
                        except Exception as e:
                            print(f"[voice-bridge] STT error: {e}", file=sys.stderr)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"[voice-bridge] Message error: {e}", file=sys.stderr)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)
        print(f"[voice-bridge] Client disconnected ({len(ws_clients)} remaining)", file=sys.stderr)


async def broadcast(message):
    """Send a message to all connected phone clients."""
    data = json.dumps(message)
    for ws in list(ws_clients):
        try:
            await ws.send(data)
        except Exception:
            ws_clients.discard(ws)


async def route_to_nats(entry):
    """Route a message to Vera's NATS inbox."""
    try:
        import nats as nats_module
        nc = await nats_module.connect(NATS_SERVER)
        await nc.publish(
            "agent.vera.inbox",
            json.dumps(entry).encode()
        )
        await nc.drain()
    except Exception as e:
        print(f"[voice-bridge] NATS route failed: {e}", file=sys.stderr)


# ── Response polling (check for Vera's replies) ──────────

async def poll_responses():
    """Poll for responses from Vera to send to phone clients."""
    response_file = Path("/opt/vera-tamago/voice_responses.jsonl")
    last_size = 0

    while True:
        await asyncio.sleep(2)

        if not response_file.exists():
            continue

        current_size = response_file.stat().st_size
        if current_size <= last_size:
            continue

        try:
            with open(response_file) as f:
                f.seek(last_size)
                new_lines = f.read()

            for line in new_lines.strip().split("\n"):
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    await broadcast({
                        "type": "voice_response",
                        "text": entry.get("content", ""),
                    })
                except json.JSONDecodeError:
                    continue

            last_size = current_size
        except Exception as e:
            print(f"[voice-bridge] Response poll error: {e}", file=sys.stderr)


# ── HTTP server for client.html ──────────────────────────

pending_responses = []   # for WebSocket broadcast
http_pending = []        # for HTTP poll delivery

class ClientHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(CLIENT_DIR), **kwargs)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/pending":
            msgs = list(http_pending)
            http_pending.clear()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"messages": msgs}).encode())
            if msgs:
                print(f"[voice-bridge] Polled: {len(msgs)} message(s) delivered", file=sys.stderr)
            return
        if self.path == "/" or self.path == "":
            self.path = "/client.html"
        super().do_GET()

    def do_POST(self):
        if self.path == "/tts":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode()
            try:
                data = json.loads(body)
                text = data.get("text", "")
                if text:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        tmpfile = f.name
                    result = subprocess.run(
                        ["/opt/vera-dreamer/venv/bin/edge-tts",
                         "--voice", VERA_VOICE,
                         f"--pitch={VERA_PITCH}",
                         "--text", text,
                         "--write-media", tmpfile],
                        capture_output=True, timeout=30
                    )
                    if result.returncode == 0:
                        with open(tmpfile, "rb") as f:
                            audio = f.read()
                        import os; os.unlink(tmpfile)
                        self.send_response(200)
                        self.send_header("Content-Type", "audio/mpeg")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(audio)
                        print(f"[voice-bridge] TTS generated: {text[:60]}", file=sys.stderr)
                        return
                    else:
                        import os; os.unlink(tmpfile)
            except Exception as e:
                print(f"[voice-bridge] TTS error: {e}", file=sys.stderr)
            self.send_error(500)
            return
        if self.path == "/reply":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode()
            try:
                data = json.loads(body)
                text = data.get("text", "")
                if text:
                    pending_responses.append(text)
                    http_pending.append(text)
                    print(f"[voice-bridge] Reply queued: {text[:80]}", file=sys.stderr)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "queued"}).encode())
                    return
            except Exception as e:
                print(f"[voice-bridge] Reply error: {e}", file=sys.stderr)
            self.send_error(400)
            return
        self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def end_headers(self):
        super().end_headers()


def start_http():
    server = HTTPServer(("0.0.0.0", HTTP_PORT), ClientHandler)
    print(f"[voice-bridge] Client UI at http://0.0.0.0:{HTTP_PORT}/", file=sys.stderr)
    server.serve_forever()


# ── Main ─────────────────────────────────────────────────

async def main():
    # Start HTTP in background thread
    http_thread = threading.Thread(target=start_http, daemon=True)
    http_thread.start()

    # Start pending response broadcaster
    async def broadcast_pending():
        while True:
            await asyncio.sleep(0.5)
            while pending_responses:
                text = pending_responses.pop(0)
                await broadcast({
                    "type": "voice_response",
                    "text": text,
                })
                print(f"[voice-bridge] Broadcast: {text[:80]}", file=sys.stderr)

    asyncio.create_task(broadcast_pending())

    # Start WebSocket server
    print(f"[voice-bridge] WebSocket on ws://0.0.0.0:{WS_PORT}", file=sys.stderr)
    async with websockets.serve(ws_handler, "0.0.0.0", WS_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    print("[voice-bridge] Vera Voice Bridge starting...", file=sys.stderr)
    asyncio.run(main())
