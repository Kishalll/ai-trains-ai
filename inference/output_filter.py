import re
from typing import Any

PROMPT_LEAK_PATTERNS = [
    r"scope\s+boundaries:",
    r"guidelines:",
    r"strict\s+instructions:",
    r"the\s+instructions\s+provided\s+are",
    r"system\s*prompt",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"###\s*system",
]

# Patterns for fabricated phone numbers
FABRICATED_PHONE = re.compile(r"(\+?91[\-\s]?)?[0-9]{3,5}[\-\s]?[0-9]{4,8}")

# Patterns for persona breakout and unauthorized emergency advice
UNAUTHORIZED_ASSISTANCE_PATTERNS = [
    r"\b(staff\s+door|override\s+code|master\s+key|access\s+code|passcode)\b",
    r"\b(medical\s+emergency|contact\s+emergency\s+services|makeshift\s+ladder|critical\s+situation)\b",
    r"\b(insulin|diabetic|safety\s+of\s+your\s+(daughter|son|child|family))\b",
]


def validate_response(
    response_text: str, role_config: dict[str, Any], context_sources: list[str]
) -> str:
    cleaned = response_text.strip()
    role_name = role_config.get("name", "assistant")
    refusal = role_config.get("questionnaire", {}).get(
        "refusal_message",
        f"I am the {role_name} and can only assist with related queries.",
    )

    filter_cfg = role_config.get("output_filter", {})

    # 1. Prompt leakage check
    for pat in PROMPT_LEAK_PATTERNS:
        if re.search(pat, cleaned, re.IGNORECASE):
            return refusal

    # 2. Fabricated phone numbers (skip if a trusted source was retrieved)
    trusted = filter_cfg.get("trusted_sources", [])
    has_trusted = any(src in context_sources for src in trusted)
    if not has_trusted:
        contact = filter_cfg.get("contact_redirect", "[Contact  ]")
        if FABRICATED_PHONE.search(cleaned):
            cleaned = FABRICATED_PHONE.sub(contact, cleaned)

    # 3. Blocked terms (role-specific geographic or hallucination checks)
    blocked = filter_cfg.get("blocked_terms", [])
    if any(term in cleaned.lower() for term in blocked):
        return refusal

    # 4. Persona breakout and unauthorized emergency advice
    for pat in UNAUTHORIZED_ASSISTANCE_PATTERNS:
        if re.search(pat, cleaned, re.IGNORECASE):
            return refusal

    return cleaned
