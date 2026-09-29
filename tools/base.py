from abc import ABC, abstractmethod
import json
from typing import Any, Optional

try:
    from jinja2 import Template
except ImportError:
    Template = None


class Tool(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[list[dict[str, Any]]] = None,
        response_template: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or []
        self.response_template = response_template

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> Any:
        """Run tool logic and return raw result data."""
        pass

    def render(self, raw_data: Any) -> str:
        """Format raw result data into text using response_template if available."""
        if not self.response_template:
            if isinstance(raw_data, (dict, list)):
                return json.dumps(raw_data, indent=2)
            return str(raw_data)

        if Template is None:
            return str(raw_data)

        try:
            tmpl = Template(self.response_template)
            # Pass data as both 'results' and 'response' for template flexibility
            rendered = tmpl.render(results=raw_data, response=raw_data)
            return rendered.strip()
        except Exception as e:
            return f"Error formatting tool output: {e}\nRaw data: {raw_data}"

    def get_signature(self) -> dict[str, Any]:
        """Return schema representation for prompt injection."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
