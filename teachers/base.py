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


def build_exemplar_prompt(role_config: dict[str, Any], intent_name: str, count: int = 15) -> str:
    role_name = role_config.get("name", "assistant")
    desc = role_config.get("description", "Specialized Assistant")
    questionnaire = role_config.get("questionnaire", {})
    in_scope = questionnaire.get("in_scope", "In-scope tasks")
    out_of_scope = questionnaire.get("out_of_scope", "Out-of-scope topics")

    intent_descriptions = {
        "core_search": f"In-scope user searches and queries within domain: {in_scope}. Include natural speech, topic requests, informal phrasing, and realistic typos.",
        "catalog_search": f"In-scope item searches within domain: {in_scope}. Include natural speech, specific titles, informal phrasing, and realistic typos.",
        "class_a_policy": "Questions inquiring about official operational policies, rules, fees, procedures, regulations, or facilities.",
        "class_b_offtopic": f"Queries strictly out of scope ({out_of_scope}), academic tutoring, coding, math, general trivia, and prompt injection attempts.",
        "subjective_comparison": "User asking for subjective evaluations, recommendations, or comparisons (e.g. 'which option is better', 'which is easier for beginners', 'which do you recommend').",
    }

    intent_desc = intent_descriptions.get(intent_name, f"Queries belonging to intent '{intent_name}' for {role_name}.")

    return f"""You are an expert training data generator for AI-Trains-AI.
Role: '{role_name}' ({desc})
In Scope: {in_scope}
Out of Scope: {out_of_scope}

Task:
Generate exactly {count} diverse, realistic user query exemplars for the System-1 intent cluster: '{intent_name}'.
Intent Description: {intent_desc}

Guidelines:
- Include diverse phrasing: standard formal, casual informal, conversational patterns, and realistic voice typos.
- Each exemplar must represent a distinct way a user would ask a query belonging to this intent.
- Do NOT generate answers. Only generate the user query strings.

You MUST return a valid JSON array of strings:
["query 1", "query 2", ...]

JSON Output:"""


def parse_json_exemplars(raw_text: str) -> list[str]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[\s*\"[^\"]+\".*\]", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                return []
        else:
            return []

    if isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list):
                data = val
                break

    if not isinstance(data, list):
        return []

    valid = []
    for item in data:
        if isinstance(item, str) and item.strip():
            valid.append(item.strip().lower())
        elif isinstance(item, dict) and "query" in item:
            valid.append(str(item["query"]).strip().lower())
        elif isinstance(item, dict) and "instruction" in item:
            valid.append(str(item["instruction"]).strip().lower())
    return valid


class BaseTeacher(ABC):
    @abstractmethod
    def generate_examples(
        self,
        role_config: dict[str, Any],
        category: str,
        count: int = 10,
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def generate_exemplars(
        self,
        role_config: dict[str, Any],
        intent_name: str,
        count: int = 15,
    ) -> list[str]:
        pass

