"""C5: provider overload (503) and rate limits (429) must rotate to another model, never block."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

pytest.importorskip("openai")
LLM = getattr(pytest.importorskip("ultralytics"), "LLM", None)  # lazily exported
if LLM is None:
    pytest.skip("ultralytics without the LLM class", allow_module_level=True)

from parkwatch.plates.readers import LLMPlateReader  # noqa: E402

CALLS = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body["model"]
        CALLS.append(model)
        code = {"busy-model": 503, "limited-model": 429}.get(model, 200)
        reply = {"id": "x", "object": "chat.completion", "created": 0, "model": model,
                 "choices": [{"index": 0, "finish_reason": "stop", "message": {
                     "role": "assistant", "content": '{"plate": "SBA 1234 K", "confidence": 0.97}'}}]}
        data = json.dumps(reply if code == 200 else {"error": {"message": "nope", "code": code}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_rotates_past_503_and_429():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        reader = LLMPlateReader({"model": "busy-model", "fallback_models": ["limited-model", "good-model"],
                                 "base_url": f"http://127.0.0.1:{srv.server_address[1]}/v1", "api_key_env": None,
                                 "timeout_s": 5})
        img = np.zeros((120, 200, 3), np.uint8)
        r1 = reader.read(img)
        assert r1.plate == "SBA 1234 K" and r1.source == "llm:good-model"
        assert CALLS == ["busy-model", "limited-model", "good-model"]
        r2 = reader.read(img)            # overloaded variants are cooling down: straight to the healthy one
        assert r2.plate and CALLS[-1] == "good-model" and len(CALLS) == 4
    finally:
        srv.shutdown()
