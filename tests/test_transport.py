import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from scheduler_lab.ollama import generate, request_payload


class Handler(BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, *args):
        pass

    def do_POST(self):
        json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        if self.mode == "error":
            self.wfile.write(b'{"error":"model failed"}\n')
            return
        events = [{"response": "one", "done": False}, {"response": "", "done": False},
                  {"response": "two", "done": False}, {"response": "three", "done": False}]
        if self.mode != "truncated":
            events.append({"response": "", "done": True, "eval_count": 3,
                           "eval_duration": 300000000, "prompt_eval_count": 128})
        for event in events:
            self.wfile.write((json.dumps(event) + "\n").encode())
            self.wfile.flush()


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        Handler.mode = "ok"

    def test_stream_metrics_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            result = generate(self.url, request_payload("fake", "hello", 3), path)
            self.assertEqual(result["chunks"], 3)
            self.assertEqual(result["generation_tps"], 10)
            self.assertTrue(result["chunk_token_count_match"])
            self.assertEqual(len(path.read_text().splitlines()), 5)

    def test_api_error_is_not_success(self):
        Handler.mode = "error"
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            generate(self.url, request_payload("fake", "hello", 3))

    def test_truncated_stream_fails(self):
        Handler.mode = "truncated"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "partial.jsonl"
            with self.assertRaisesRegex(RuntimeError, "without a done"):
                generate(self.url, request_payload("fake", "hello", 3), path)
            self.assertEqual(len(path.read_text().splitlines()), 4)
