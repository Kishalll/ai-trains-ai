import json
import os
from typing import Any, Optional
import urllib.request

from teachers.base import BaseTeacher, build_teacher_prompt, parse_json_examples

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAITeacher(BaseTeacher):
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

    def generate_examples(
        self,
        role_config: dict[str, Any],
        category: str,
        count: int = 10,
    ) -> list[dict[str, Any]]:
        prompt = build_teacher_prompt(role_config, category, count)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert training data generator. Output only valid JSON arrays.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "top_p": 0.9,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

        return parse_json_examples(content, category)
