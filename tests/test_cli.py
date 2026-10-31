"""Tests for the CLI layer (mocked end to end, no keys)."""

from __future__ import annotations

import unittest
from unittest import mock

from consumer_complaint_drafter.cli import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    build_parser,
    collect_details,
    draft_letter,
    main,
)


class CliParserTests(unittest.TestCase):
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["complaint"])
        self.assertEqual(args.type, "complaint")
        self.assertEqual(args.endpoint, DEFAULT_ENDPOINT)
        self.assertEqual(args.model, DEFAULT_MODEL)
        self.assertIsNone(args.output)

    def test_custom_endpoint_and_model(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--endpoint",
                "http://127.0.0.1:9999/v1",
                "--model",
                "mistral",
                "--timeout",
                "120",
                "--output",
                "/tmp/letter.txt",
                "warranty",
            ]
        )
        self.assertEqual(args.endpoint, "http://127.0.0.1:9999/v1")
        self.assertEqual(args.model, "mistral")
        self.assertEqual(args.timeout, 120.0)
        self.assertEqual(args.output, "/tmp/letter.txt")
        self.assertEqual(args.type, "warranty")

    def test_all_types_accepted(self):
        parser = build_parser()
        for letter_type in ("complaint", "warranty", "small-claims"):
            args = parser.parse_args([letter_type])
            self.assertEqual(args.type, letter_type)


class CollectDetailsTests(unittest.TestCase):
    def test_collect_details_prompts_all_fields(self):
        answers = iter(
            [
                "Ada Tester",
                "123 Test St, Indianapolis IN 46201",
                "ada@example.com",
                "555-0100",
                "",  # date -> default
                "Widget Co",
                "456 Merchant Ave",
                "ACC-999",
                "Widget 3000",
                "2026-01-01",
                "It arrived broken.",
                "Called support twice.",
                "Full refund.",
                "2026-11-15",
                "",
                "",
                "",
                "",
            ]
        )
        with mock.patch("builtins.input", side_effect=lambda prompt="": next(answers)):
            details = collect_details("complaint")
        self.assertEqual(details.letter_type, "complaint")
        self.assertEqual(details.your_name, "Ada Tester")
        self.assertEqual(details.problem_description, "It arrived broken.")
        self.assertEqual(details.desired_resolution, "Full refund.")


class DraftLetterTests(unittest.TestCase):
    def test_draft_letter_calls_client_and_returns_content(self):
        details = mock.Mock()
        details.letter_type = "complaint"
        client = mock.Mock()
        client.complete.return_value.content = "DRAFT TEXT"
        result = draft_letter(details, client)
        self.assertEqual(result, "DRAFT TEXT")
        client.complete.assert_called_once()


class MainTests(unittest.TestCase):
    def test_main_prints_letter_and_disclaimer(self):
        with mock.patch(
            "consumer_complaint_drafter.cli.collect_details"
        ) as collect, mock.patch(
            "consumer_complaint_drafter.cli.draft_letter"
        ) as draft, mock.patch("builtins.print") as print_mock:
            collect.return_value = mock.Mock(letter_type="complaint")
            draft.return_value = "THE LETTER BODY"
            code = main(["complaint"])
        self.assertEqual(code, 0)
        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
        self.assertIn("THE LETTER BODY", printed)
        self.assertIn("EDUCATIONAL PURPOSES ONLY", printed)
        self.assertIn("NEVER sends letters automatically", printed)

    def test_main_saves_to_file_with_output_flag(self):
        with mock.patch(
            "consumer_complaint_drafter.cli.collect_details"
        ) as collect, mock.patch(
            "consumer_complaint_drafter.cli.draft_letter"
        ) as draft, mock.patch("builtins.open", mock.mock_open()) as open_mock:
            collect.return_value = mock.Mock(letter_type="warranty")
            draft.return_value = "WARRANTY DRAFT"
            code = main(["warranty", "--output", "/tmp/saved.txt"])
        self.assertEqual(code, 0)
        open_mock.assert_called_once_with("/tmp/saved.txt", "w", encoding="utf-8")
        handle = open_mock()
        handle.write.assert_called()
        written = handle.write.call_args[0][0]
        self.assertIn("WARRANTY DRAFT", written)
        self.assertIn("EDUCATIONAL PURPOSES ONLY", written)

    def test_main_endpoint_error_returns_nonzero(self):
        with mock.patch(
            "consumer_complaint_drafter.cli.collect_details",
            side_effect=RuntimeError("boom"),
        ):
            # collect_details raising EndpointError is what main handles
            from consumer_complaint_drafter.client import EndpointError

            with mock.patch(
                "consumer_complaint_drafter.cli.collect_details",
                side_effect=EndpointError("down"),
            ), mock.patch("builtins.print"):
                code = main(["complaint"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
