#!/usr/bin/env python3
"""
Simple proxy server for the Agent Demo.
Keeps your API key on the server side, away from the browser.

Usage:
  1. pip install anthropic
  2. Set your API key below (or as environment variable ANTHROPIC_API_KEY)
  3. python server.py
  4. Open http://localhost:8080 in your browser
"""

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ─── SET YOUR API KEY HERE ───────────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
# ─────────────────────────────────────────────────────────────────────────────

client = Anthropic(api_key=API_KEY)

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"  {args[0]} {args[1]}")

    def send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        # Serve the demo HTML file
        try:
            with open("agent-demo-local.html", "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_cors()
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"agent-demo-local.html not found in same directory")

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        try:
            response = client.messages.create(
                model=body.get("model", "claude-sonnet-4-20250514"),
                max_tokens=body.get("max_tokens", 1000),
                system=body.get("system", ""),
                messages=body.get("messages", [])
            )
            result = {"content": [{"type": "text", "text": response.content[0].text}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            error = {"error": str(e)}
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_cors()
            self.end_headers()
            self.wfile.write(json.dumps(error).encode())


if __name__ == "__main__":
    port = 8080
    print(f"\n  Agent Demo Server")
    print(f"  ─────────────────")
    print(f"  Running at → http://localhost:{port}")
    print(f"  API key    → {'set ✓' if API_KEY != 'YOUR_API_KEY_HERE' else 'NOT SET ✗ — edit server.py'}")
    print(f"\n  Press Ctrl+C to stop\n")
    HTTPServer(("", port), Handler).serve_forever()
