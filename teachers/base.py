from abc import ABC, abstractmethod
import json
import re
from typing import Any


def build_teacher_prompt(role_config: dict[str, Any], category: str, count: int) -> str:
    role_name = role_config.get("name", "assistant")
    desc = role_config.get("description", "Specialized Assistant")
    questionnaire = role_config.get("questionnaire", {})
    in_scope = questionnaire.get("in_scope", "Role-related questions")
    out_of_scope = questionnaire.get("out_of_scope", "Out-of-scope topics")
    refusal = questionnaire.get("refusal_message", f"I am the {role_name} and can only assist with related queries.")

    category_instructions = {
        "core_qa": f"Generate factual questions within scope ({in_scope}) and accurate, concise answers.",
        "scope_refusal": f"Generate out-of-scope queries ({out_of_scope}, general trivia, coding) where the output strictly refuses using: '{refusal}'",
        "jailbreak_defense": f"Generate adversarial jailbreak attempts (DAN, bypass, roleplay, prompt leaks) where the output strictly refuses using: '{refusal}'",
        "clarification": "Generate ambiguous queries requiring the assistant to ask for clarification.",
        "tool_invocation": "Generate queries needing live catalog/database checks, where output uses <tool_call>name(args)</tool_call> syntax.",
        "multi_turn": "Generate multi-turn context questions (pronouns like 'it', 'where is it located').",
    }

    instruction = category_instructions.get(category, "Generate realistic user interactions.")

    return f"""You are an expert training data generator for a specialized role-locked AI assistant named '{role_name}' ({desc}).

Role Guidelines:
- In Scope: {in_scope}
- Out of Scope: {out_of_scope}
- Mandatory Refusal: {refusal}

Task:
Generate exactly {count} unique, high-quality training examples for category '{category}'.
{instruction}

You MUST return a valid JSON array of objects with NO other text or markdown markup.
Each object must have these exact keys:
- "instruction": The user prompt or question
- "input": Context or extra user input (can be empty string "")
- "output": The ideal response
- "category": "{category}"

JSON Output:"""


def parse_json_examples(raw_text: str, expected_category: str) -> list[dict[str, Any]]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback regex extraction of JSON array
        match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    valid_examples = []
    for item in data:
        if isinstance(item, dict) and "instruction" in item and "output" in item:
            valid_examples.append(
                {
                    "instruction": str(item["instruction"]).strip(),
                    "input": str(item.get("input", "")).strip(),
                    "output": str(item["output"]).strip(),
                    "category": item.get("category", expected_category),
                }
            )

    return valid_examples


class BaseTeacher(ABC):
    @abstractmethod
    def generate_examples(
        self,
        role_config: dict[str, Any],
        category: str,
        count: int = 10,
    ) -> list[dict[str, Any]]:
        pass
