#!/usr/bin/env python3
"""
Whisper STT endpoint for Vera's voice channel.
Runs on MTH. Accepts audio via HTTP POST, returns transcript.

POST /transcribe
  Body: audio file (webm, wav, mp3, etc)
  Returns: {"text": "transcribed text"}

GET /health
  Returns: {"status": "ready", "model": "small.en"}
"""

import io
import sys
import tempfile
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

from faster_whisper import WhisperModel

MODEL_SIZE = "small.en"
HOST = "0.0.0.0"
PORT = 8766

print(f"Loading Whisper {MODEL_SIZE}...", file=sys.stderr)
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Whisper ready.", file=sys.stderr)


class STTHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/transcribe":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0 or content_length > 10_000_000:
            self.send_error(400, "Audio too large or empty")
            return

        audio_data = self.rfile.read(content_length)

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_data)
            tmpfile = f.name

        try:
            segments, _ = model.transcribe(
                tmpfile,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

            # Apply word fixes
            fixes = {
                "brodice": "Bright-Eyes",
                "bright eyes": "Bright-Eyes",
                "bright-ice": "Bright-Eyes",
                "brodise": "Bright-Eyes",
            }
            text_lower = text.lower()
            for wrong, right in fixes.items():
                if wrong in text_lower:
                    import re
                    text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
                    text_lower = text.lower()

            response = json.dumps({"text": text})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response.encode())
            print(f"[stt] Transcribed: {text}", file=sys.stderr)

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            print(f"[stt] Error: {e}", file=sys.stderr)
        finally:
            os.unlink(tmpfile)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready", "model": MODEL_SIZE}).encode())
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), STTHandler)
    print(f"Whisper STT listening on {HOST}:{PORT}", file=sys.stderr)
    server.serve_forever()
