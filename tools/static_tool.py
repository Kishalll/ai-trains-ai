from typing import Any, Optional

from tools.base import Tool


class StaticTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        response: str,
        parameters: Optional[list[dict[str, Any]]] = None,
        response_template: Optional[str] = None,
    ):
        super().__init__(name, description, parameters, response_template)
        self.static_response = response.strip()

    def execute(self, arguments: dict[str, Any]) -> str:
        return self.static_response
