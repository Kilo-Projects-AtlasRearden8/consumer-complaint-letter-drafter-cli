"""Letter-type prompt templates used to instruct the language model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LetterDetails:
    """User-supplied facts used to build a letter draft."""

    letter_type: str
    your_name: str
    your_address: str
    your_email: str = ""
    your_phone: str = ""
    date: str = ""
    recipient_name: str = ""
    recipient_address: str = ""
    account_number: str = ""
    product_or_service: str = ""
    date_of_purchase: str = ""
    problem_description: str = ""
    steps_taken: str = ""
    desired_resolution: str = ""
    response_by: str = ""
    small_claims_amount: str = ""
    court_name: str = ""
    claim_number: str = ""
    warranty_type: str = ""


LETTER_SYSTEM_PROMPT = (
    "You are a careful, plain-English writing assistant. Draft a professional, "
    "respectful, concise business letter based only on the facts the user "
    "provides. Do not invent facts, dates, amounts, or legal claims. Do not "
    "give legal advice. Use a firm but polite tone. Include a header block with "
    "sender and recipient addresses and a date, a subject line, a clear body, "
    "a requested response deadline, and a signature block."
)


def letter_prompt(details: LetterDetails) -> str:
    """Build the chat prompt for the requested letter type."""
    common = (
        f"Sender: {details.your_name}\n"
        f"Sender address: {details.your_address}\n"
        f"Sender email: {details.your_email}\n"
        f"Sender phone: {details.your_phone}\n"
        f"Date: {details.date}\n"
        f"Recipient: {details.recipient_name}\n"
        f"Recipient address: {details.recipient_address}\n"
        f"Account number: {details.account_number}\n"
        f"Product or service: {details.product_or_service}\n"
        f"Date of purchase: {details.date_of_purchase}\n"
        f"Problem: {details.problem_description}\n"
        f"Steps already taken: {details.steps_taken}\n"
        f"Desired resolution: {details.desired_resolution}\n"
        f"Requested response by: {details.response_by}\n"
    )

    if details.letter_type == "complaint":
        return (
            "Write a consumer complaint letter to a business about a product or "
            "service problem. Use these facts:\n" + common
        )
    if details.letter_type == "warranty":
        return (
            "Write a warranty claim letter requesting repair, replacement, or "
            "refund under a warranty. Use these facts:\n" + common
        )
    if details.letter_type == "small-claims":
        return (
            "Write a formal small-claims demand letter demanding payment or "
            "action before filing in small-claims court. Include the amount "
            "claimed, court, and claim number if provided. Use these facts:\n"
            + common
        )

    raise ValueError(f"Unknown letter type: {details.letter_type!r}")


LETTER_TYPES = ("complaint", "warranty", "small-claims")
