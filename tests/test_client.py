"""Tests for consumer_complaint_drafter using a mock endpoint (no keys)."""

from __future__ import annotations

import json
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest import mock

from consumer_complaint_drafter.client import (
    CompletionClient,
    CompletionRequest,
    EndpointError,
)
from consumer_complaint_drafter.disclaimer import render_disclaimer
from consumer_complaint_drafter.letters import (
    LETTER_TYPES,
    LetterDetails,
    letter_prompt,
)


class MockCompletionHandler(BaseHTTPRequestHandler):
    """HTTP handler that returns a fixed completion for any POST."""

    # Class-level capture of the last request body for inspection.
    last_request_body: bytes = b""

    def do_POST(self):  # noqa: N802 - http.server naming convention
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        MockCompletionHandler.last_request_body = body
        payload = {
            "model": "llama3.2",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "MOCKED LETTER DRAFT CONTENT",
                    }
                }
            ],
        }
        resp = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *args):  # noqa: D401 - silence test server logs
        pass


class MockEndpointTests(unittest.TestCase):
    """Integration-style tests against a live local mock HTTP server."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockCompletionHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_client_reaches_mock_endpoint(self):
        client = CompletionClient(
            base_url=f"http://127.0.0.1:{self.port}/v1",
            model="mock-model",
        )
        response = client.complete(
            CompletionRequest(
                model="mock-model",
                messages=[{"role": "user", "content": "draft a letter"}],
            )
        )
        self.assertEqual(response.content, "MOCKED LETTER DRAFT CONTENT")
        self.assertEqual(response.model, "llama3.2")

    def test_client_model_falls_back_to_client_default(self):
        client = CompletionClient(
            base_url=f"http://127.0.0.1:{self.port}/v1",
            model="default-model",
        )
        response = client.complete(
            CompletionRequest(model="", messages=[{"role": "user", "content": "hi"}])
        )
        self.assertEqual(response.content, "MOCKED LETTER DRAFT CONTENT")

    def test_client_sends_correct_payload_to_endpoint(self):
        client = CompletionClient(
            base_url=f"http://127.0.0.1:{self.port}/v1",
            model="sent-model",
        )
        client.complete(
            CompletionRequest(
                model="sent-model",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "user"},
                ],
                temperature=0.5,
                max_tokens=100,
            )
        )
        sent = json.loads(MockCompletionHandler.last_request_body.decode("utf-8"))
        self.assertEqual(sent["model"], "sent-model")
        self.assertEqual(
            sent["messages"], [{"role": "system", "content": "sys"},
                               {"role": "user", "content": "user"}]
        )
        self.assertEqual(sent["temperature"], 0.5)
        self.assertEqual(sent["max_tokens"], 100)


class ClientErrorTests(unittest.TestCase):
    """Tests for client error handling without a running endpoint."""

    def test_connection_refused_raises_endpoint_error(self):
        client = CompletionClient(
            base_url="http://127.0.0.1:1", model="any", timeout_seconds=2.0
        )
        with self.assertRaises(EndpointError):
            client.complete(
                CompletionRequest(
                    model="any",
                    messages=[{"role": "user", "content": "hi"}],
                )
            )


class LetterPromptTests(unittest.TestCase):
    """Tests for letter prompt construction."""

    def test_all_letter_types_are_supported(self):
        self.assertEqual(LETTER_TYPES, ("complaint", "warranty", "small-claims"))

    def test_prompt_includes_user_facts(self):
        details = LetterDetails(
            letter_type="complaint",
            your_name="Ada Tester",
            your_address="123 Test St, IN 46201",
            product_or_service="Widget 3000",
            problem_description="It broke on first use.",
            desired_resolution="A full refund of $25.00.",
        )
        prompt = letter_prompt(details)
        self.assertIn("Ada Tester", prompt)
        self.assertIn("Widget 3000", prompt)
        self.assertIn("It broke on first use.", prompt)
        self.assertIn("full refund of $25.00", prompt)
        self.assertIn("consumer complaint letter", prompt.lower())

    def test_warranty_and_small_claims_have_type_specific_language(self):
        base = dict(
            letter_type="warranty",
            your_name="Ada Tester",
            your_address="123 Test St",
            product_or_service="Widget 3000",
            problem_description="It broke.",
            desired_resolution="Replace it.",
        )
        warranty_prompt = letter_prompt(LetterDetails(**base))
        self.assertIn("warranty claim letter", warranty_prompt.lower())

        small = letter_prompt(
            LetterDetails(**{**base, "letter_type": "small-claims"})
        )
        self.assertIn("small-claims", small.lower())

    def test_unknown_letter_type_raises(self):
        with self.assertRaises(ValueError):
            letter_prompt(
                LetterDetails(
                    letter_type="not-a-real-type",  # type: ignore[arg-type]
                    your_name="Ada Tester",
                    your_address="123 Test St",
                )
            )


class DisclaimerTests(unittest.TestCase):
    """Tests that every output carries the education disclaimer."""

    def test_disclaimer_is_present_and_educational(self):
        text = render_disclaimer()
        self.assertIn("EDUCATIONAL PURPOSES ONLY", text)
        self.assertIn("NOT legal advice", text)
        self.assertIn("NEVER sends letters automatically", text)
        self.assertIn("attorney-client relationship", text)


if __name__ == "__main__":
    unittest.main()
