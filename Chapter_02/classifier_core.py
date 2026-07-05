"""
classifier_core.py
Standalone, importable module containing PROMPT_REGISTRY and classify_email().

This exists so Chapter 5 (and anything else) can do:
    from classifier_core import PROMPT_REGISTRY, classify_email
without needing prompt_engineering_chapter4.ipynb converted to a .py file.

If you change a prompt in your Chapter 4 notebook, copy the matching change
into this file too -- the notebook and this file are two separate copies of
the same content, not linked to each other.

Setup:
    pip install -U anthropic pydantic python-dotenv
    .env file containing: ANTHROPIC_API_KEY=sk-ant-...
"""

import os
from typing import Literal

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL_ID = "claude-haiku-4-5-20251001"


# ----------------------------------------------------------------------
# Structured output schemas
# ----------------------------------------------------------------------

class EmailClassification(BaseModel):
    """Used by v0, v1, v2."""
    label: Literal["FD", "Non-FD", "Multiple Category"]
    reason: str


class EmailClassificationWithReasoning(BaseModel):
    """Used by v3 (chain-of-thought). `reasoning` is declared FIRST and is
    required, so the model is structurally forced to generate it BEFORE
    `label` -- required properties are emitted in schema order."""
    reasoning: str = Field(description="Step-by-step reasoning, written BEFORE deciding the label.")
    label: Literal["FD", "Non-FD", "Multiple Category"]
    reason: str = Field(description="One short sentence summarizing the final decision.")


# ----------------------------------------------------------------------
# Prompt versions
# ----------------------------------------------------------------------

V0_ZERO_SHOT = """You classify customer emails for Bajaj Finance into one of three categories:

- FD: the email is about a Fixed Deposit account
- Non-FD: the email is about anything else
- Multiple Category: the email is about both FD and non-FD topics
"""

# NOTE: the keyword groups and negation phrases below are ILLUSTRATIVE
# placeholders. Swap in your actual 8 keyword groups and 20-phrase negation
# list from the rule-engine cheatsheet before treating this as a real
# comparison against the classical baseline.
V1_DOMAIN_KNOWLEDGE = """You classify customer emails for Bajaj Finance into one of three categories:

- FD: the email is about a Fixed Deposit account -- maturity, interest/quarterly
  payout, premature withdrawal, rollover, or an FD reference number matching the
  pattern BJ + 4-digit year + FD + digits (e.g. BJ2019FD7717).
- Non-FD: the email is about anything else -- loans, EMIs, insurance, cards, the
  mobile app, or branch service, with no FD reference.
- Multiple Category: the email raises BOTH an FD concern AND a Non-FD concern.

Domain signal groups:
- FD-signal keywords     : maturity, interest payout, premature withdrawal, rollover, FD reference number
- Non-FD-signal keywords : EMI, loan, insurance premium, card, app login, branch visit
- Negation phrases to watch for: "nahi mila" (didn't get), "abhi tak nahi aaya"
  (still hasn't come) -- a negated FD keyword can still mean the email IS about
  FD (the customer is complaining something FD-related didn't happen), so
  don't auto-route purely on keyword presence.

Emails are often in Hinglish (Hindi written in Roman script, mixed with English).
Treat Hinglish phrasing as normal customer language, not as noise.
"""

V2_FEW_SHOT = V1_DOMAIN_KNOWLEDGE + """
Examples:

Email: "Mera paisa abhi tak nahi aaya. Kab milega? Bahut time ho gaya hai. BJ2024FD9354."
Output: {"label": "FD", "reason": "Customer is asking about money tied to FD reference BJ2024FD9354."}

Email: "Sir ji, App me login nahi ho raha. OTP aata hi nahi. Teen din se try kar raha hoon. Kya problem hai?"
Output: {"label": "Non-FD", "reason": "Complaint is about app login/OTP, no FD reference at all."}

Email: "Aapke yahan mera paisa hai BJ2022FD5397. Uska kya status hai? And separately, I want to foreclose my personal loan BJ2023CD2320."
Output: {"label": "Multiple Category", "reason": "Raises both an FD status question and a separate loan foreclosure request."}
"""

V3_CHAIN_OF_THOUGHT = V1_DOMAIN_KNOWLEDGE + """
Before deciding, reason step by step: list every FD-signal and every
Non-FD-signal phrase you find in the email, check each one for negation,
then commit to a label based on that reasoning.
"""


# ----------------------------------------------------------------------
# Prompt registry -- prompt text + schema + max_tokens + changelog,
# tracked together per version.
# ----------------------------------------------------------------------
PROMPT_REGISTRY = {
    "v0_zero_shot": {
        "system_prompt": V0_ZERO_SHOT,
        "schema": EmailClassification,
        "max_tokens": 200,
        "changelog": "Baseline. No domain knowledge, no examples. What's the floor?",
    },
    "v1_domain_knowledge": {
        "system_prompt": V1_DOMAIN_KNOWLEDGE,
        "schema": EmailClassification,
        "max_tokens": 200,
        "changelog": "Added FD reference pattern, keyword groups, negation warning, "
                      "Hinglish note. Expect fewer keyword-only misreads than v0.",
    },
    "v2_few_shot": {
        "system_prompt": V2_FEW_SHOT,
        "schema": EmailClassification,
        "max_tokens": 200,
        "changelog": "Added 3 real labeled examples (one per class) on top of v1. "
                      "Expect Multiple Category to improve most -- it's the hardest "
                      "class to describe in rules alone.",
    },
    "v3_chain_of_thought": {
        "system_prompt": V3_CHAIN_OF_THOUGHT,
        "schema": EmailClassificationWithReasoning,
        "max_tokens": 600,  # the reasoning field alone can run past 300 tokens
        "changelog": "Same domain knowledge as v1, but forces explicit reasoning "
                      "before the label via schema field ordering. Costs more output "
                      "tokens -- only worth it if v0-v2 get borderline cases wrong.",
    },
}


def classify_email(version: str, subject: str, content: str):
    """Run one email through a named prompt version. Returns the parsed
    Pydantic object directly -- no json.loads(), no fence-stripping, no
    PARSE_ERROR fallback. Structured outputs guarantee schema-valid output
    AS LONG AS the response doesn't get cut off by max_tokens first."""
    entry = PROMPT_REGISTRY[version]
    user_message = f"Subject: {subject}\n\nBody: {content}"

    try:
        response = client.messages.parse(
            model=MODEL_ID,
            max_tokens=entry["max_tokens"],
            system=entry["system_prompt"],
            messages=[{"role": "user", "content": user_message}],
            output_format=entry["schema"],
        )
    except Exception as e:
        raise RuntimeError(
            f"[{version}] structured output failed to parse -- this usually means "
            f"the response was cut off before the JSON closed. Try raising "
            f"max_tokens for this version (currently {entry['max_tokens']}). "
            f"Original error: {e}"
        ) from e

    return response.parsed_output