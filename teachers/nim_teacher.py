import json
import os
from typing import Any, Optional
import urllib.request

from teachers.base import BaseTeacher, build_teacher_prompt, parse_json_examples

DEFAULT_NIM_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NIM_MODEL = "meta/llama-3.1-70b-instruct"


class NimTeacher(BaseTeacher):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name or os.environ.get("NVIDIA_MODEL", DEFAULT_NIM_MODEL)
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_API_KEY")
        self.base_url = (base_url or os.environ.get("NVIDIA_BASE_URL", DEFAULT_NIM_URL)).rstrip("/")

        if not self.api_key and not self.base_url.startswith("http://localhost"):
            raise ValueError(
                "NVIDIA API Key not found. Please set NVIDIA_API_KEY or NIM_API_KEY environment variable."
            )

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
                    "content": "You are a professional dataset generator. Output ONLY valid JSON arrays.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
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
