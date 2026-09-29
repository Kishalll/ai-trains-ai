import json
from typing import Any
import urllib.request

from teachers.base import BaseTeacher, build_teacher_prompt, parse_json_examples


class OllamaTeacher(BaseTeacher):
    def __init__(self, model_name: str = "qwen2.5:1.5b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate_examples(
        self,
        role_config: dict[str, Any],
        category: str,
        count: int = 5,
    ) -> list[dict[str, Any]]:
        prompt = build_teacher_prompt(role_config, category, count)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            },
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "")

        return parse_json_examples(content, category)
