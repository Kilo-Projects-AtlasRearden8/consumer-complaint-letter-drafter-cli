"""Command-line interface for drafting consumer letters."""

from __future__ import annotations

import argparse
import sys
from datetime import date

from .client import CompletionClient, CompletionRequest, EndpointError
from .disclaimer import render_disclaimer
from .letters import LETTER_TYPES, LetterDetails, letter_prompt

DEFAULT_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2"
DEFAULT_TIMEOUT = 60.0


def _prompt(field: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{field}{suffix}: ").strip()
    return value or default


def collect_details(letter_type: str) -> LetterDetails:
    """Interactively collect the facts needed for a letter draft."""
    print(f"\nDrafting a {letter_type} letter. Press Enter to skip optional fields.\n")
    return LetterDetails(
        letter_type=letter_type,
        your_name=_prompt("Your name"),
        your_address=_prompt("Your address (one line)"),
        your_email=_prompt("Your email (optional)"),
        your_phone=_prompt("Your phone (optional)"),
        date=_prompt("Date", date.today().isoformat()),
        recipient_name=_prompt("Company or recipient name"),
        recipient_address=_prompt("Recipient address (optional)"),
        account_number=_prompt("Account or reference number (optional)"),
        product_or_service=_prompt("Product or service"),
        date_of_purchase=_prompt("Date of purchase or service (optional)"),
        problem_description=_prompt("Describe the problem"),
        steps_taken=_prompt("Steps you have already taken (optional)"),
        desired_resolution=_prompt("What resolution do you want (refund, repair, etc.)"),
        response_by=_prompt("Request a response by (date, optional)"),
        small_claims_amount=_prompt("Amount you are claiming (small-claims only, optional)"),
        court_name=_prompt("Court name (small-claims only, optional)"),
        claim_number=_prompt("Claim number (small-claims only, optional)"),
        warranty_type=_prompt("Warranty type (warranty claims only, optional)"),
    )


def draft_letter(details: LetterDetails, client: CompletionClient) -> str:
    """Ask the language model to draft the letter and return the text."""
    messages = [
        {"role": "system", "content": letter_prompt(details)},
        {
            "role": "user",
            "content": "Write the letter now using only the facts above.",
        },
    ]
    response = client.complete(
        CompletionRequest(
            model=client.model,
            messages=messages,
            temperature=0.3,
        )
    )
    return response.content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="consumer-complaint-drafter",
        description=(
            "Draft consumer complaint, warranty, and small-claims demand "
            "letters with a local OpenAI-compatible model (Ollama by default). "
            "Educational drafts only; this tool never sends anything."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="consumer-complaint-drafter 0.1.0",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Base URL of the OpenAI-compatible endpoint (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Save the drafted letter to this file instead of printing it",
    )
    parser.add_argument(
        "type",
        choices=LETTER_TYPES,
        help="Type of letter to draft",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    client = CompletionClient(
        base_url=args.endpoint,
        model=args.model,
        timeout_seconds=args.timeout,
    )

    try:
        details = collect_details(args.type)
        letter = draft_letter(details, client)
    except EndpointError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Tip: start Ollama (`ollama serve`), pull a model "
            "(`ollama pull llama3.2`), or point --endpoint at another "
            "OpenAI-compatible server.",
            file=sys.stderr,
        )
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.", file=sys.stderr)
        return 130

    output = letter + "\n\n" + render_disclaimer()

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(output)
        except OSError as exc:
            print(f"Error writing {args.output}: {exc}", file=sys.stderr)
            return 1
        print(f"Letter draft saved to {args.output}")
        print(render_disclaimer())
        return 0

    print("\n================================================================================")
    print("DRAFTED LETTER")
    print("================================================================================\n")
    print(letter)
    print("\n" + render_disclaimer())
    print(
        "\nReview, edit, and send this letter yourself. This tool never sends anything."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
