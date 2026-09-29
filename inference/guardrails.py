import re
from typing import Any, Optional

# Unicode ranges for South Asian scripts
SCRIPT_PATTERNS = {
    "tamil": re.compile(r"[\u0B80-\u0BFF]"),
    "malayalam": re.compile(r"[\u0D00-\u0D7F]"),
    "telugu": re.compile(r"[\u0C00-\u0C7F]"),
    "kannada": re.compile(r"[\u0C80-\u0CFF]"),
    "devanagari": re.compile(r"[\u0900-\u097F]"),
}

# Transliterated regional words
ROMANIZED_REGIONAL_WORDS = {
    "hindi": ["kaise", "kya", "hai", "karo", "batao", "chahiye", "kripya", "samay", "bhai", "nahi", "namaste"],
    "tamil": ["eppo", "eppadi", "theriyuma", "irukka", "venum", "kudunga", "neram", "illaya", "aagum", "solla", "vanakkam", "thambi"],
    "malayalam": ["eppozhanu", "enthanu", "undo", "kittumo", "parayamo", "samayam", "illa", "namaskaram", "chechi", "chetta"],
    "telugu": ["eppudu", "ela", "undi", "kavali", "cheppandi", "ledu", "namaskaram", "anna"],
    "kannada": ["yaavaga", "hege", "ideya", "beku", "heli", "illa", "namaskara"],
}

# Common jailbreak and persona bypass patterns
JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|given|system)?\s*(instructions|constraints|rules|prompts|guidelines|boundaries)",
    r"you\s+are\s+(now\s+)?(in\s+)?(an?\s+)?(unrestricted|dan|developer\s+mode|jailbroken)",
    r"act\s+as\s+(an?\s+)?(unrestricted|dan|hacker|root|admin)",
    r"pretend\s+(you\s+have\s+no\s+rules|you\s+are\s+unfiltered|to\s+be)",
    r"bypass\s+(safety|rules|guardrails|filters|constraints)",
    r"system\s*prompt",
    r"###\s*system",
    r"\[inst\]",
    r"<\|im_start\|>",
    r"\b(admin|system|root|diagnostic)\s+(test|command|check|verification|mode|override)\b",
    r"\breveal\s+(who\s+you\s+(actually\s+)?are|your\s+(real\s+)?(identity|instructions|system|prompt))\b",
    r"\b(roleplay|role-play|role\s+playing)\b",
    r"\b(simulate|simulation)\b",
    r"\bimagine\s+(you('re|\s+are)?|your|yourself)\b",
    r"\bplay\s+a\s+(simulation|game|role|scene)\b",
    r"\b(forget|stop\s+being|drop\s+the|you('re|\s+are)\s+not)\s+(that\s+)?(you('re|\s+are)\s+)?(a\s+|an\s+)?(librarian|assistant|bot|ai|role)\b",
    r"\b(not\s+(asking\s+you\s+to\s+do\s+your\s+job|a\s+library\s+question|a\s+librarian\s+function))\b",
    r"\b(more\s+than\s+(a\s+)?(policy|librarian|bot|system))\b",
    r"\b(override\s+code|door\s+code|master\s+code|access\s+code|passcode|reset\s+code|lock\s+code|staff\s+door|security\s+override)\b",
    r"\b(camera\s+ip|surveillance\s+(camera|feed|log|ip)|cctv\s+feed)\b",
    r"\b(crime\s+branch|police\s+(station|inspector|officer)|section\s+144|investigative\s+powers)\b",
    r"\b(lives\s+are\s+at\s+stake|real\s+emergency|life\s+or\s+death|medical\s+emergency|insulin|diabetic|heart\s+attack|stroke|bleeding\s+out)\b",
    r"\b(asthma|inhaler|seizure|choking|tracheostomy|anaphylaxis|epipen|antivenom|cpr|defuse)\b",
]


def check_input_guardrails(
    message: str, role_config: dict[str, Any]
) -> tuple[bool, Optional[str]]:
    text = message.strip()
    role_name = role_config.get("name", "assistant")
    refusal = role_config.get("questionnaire", {}).get(
        "refusal_message",
        f"I am the {role_name} and can only assist with related queries.",
    )

    # 1. Context stuffing / length cap
    if len(text) > 1000:
        return False, "Input exceeds maximum allowed length (1000 characters). Please ask a shorter question."

    # 2. Non-English script detection
    for script_name, pattern in SCRIPT_PATTERNS.items():
        if pattern.search(text):
            return False, "I can only assist in English. Could you please ask your question in English?"

    # 3. Transliterated non-English detection
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    for lang, markers in ROMANIZED_REGIONAL_WORDS.items():
        matches = [m for m in markers if m in tokens]
        if len(matches) >= 1:
            return False, "I can only assist in English. Could you please ask your question in English?"

    # 4. Jailbreak pattern detection
    for pat in JAILBREAK_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return False, refusal

    # 5. Off-topic pattern detection (loaded from role config)
    off_topic = role_config.get("guardrails", {}).get("off_topic_patterns", [])
    for pat in off_topic:
        if re.search(pat, text, re.IGNORECASE):
            return False, refusal

    return True, None
