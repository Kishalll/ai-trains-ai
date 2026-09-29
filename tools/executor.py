from collections import defaultdict
import json
import re
from typing import Any, Optional

from tools.base import Tool


class ToolExecutor:
    def __init__(self, tools: dict[str, Tool], max_calls_per_session: int = 10):
        self.tools = tools
        self.max_calls_per_session = max_calls_per_session
        self.session_counts: dict[str, int] = defaultdict(int)

    def execute(self, tool_name: str, arguments: dict[str, Any], session_id: Optional[str] = None) -> str:
        sid = session_id or "default"
        if self.session_counts[sid] >= self.max_calls_per_session:
            return "Tool call limit reached for this session."
        self.session_counts[sid] += 1

        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found."

        try:
            raw_result = tool.execute(arguments)
            return tool.render(raw_result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"

    def parse_tool_calls(self, text: str) -> list[tuple[str, dict[str, Any]]]:
        """Extract tool calls from <tool_call>...</tool_call> tags."""
        pattern = r"<tool_call>(.*?)(?:</tool_call>|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        calls = []

        for m in matches:
            content = m.strip()
            if not content:
                continue

            # Format 1: JSON payload
            if content.startswith("{"):
                try:
                    # Clean any trailing garbage
                    json_str = content
                    if not json_str.endswith("}"):
                        json_str = json_str[:json_str.rfind("}") + 1] if "}" in json_str else json_str + "}"
                    data = json.loads(json_str)
                    t_name = data.get("name")
                    t_args = data.get("arguments", {})
                    if t_name:
                        calls.append((t_name, t_args))
                        continue
                except json.JSONDecodeError:
                    pass

            # Format 2: Function call style name(arg1="val", arg2="val") or name("val")
            fn_match = re.match(r"^([a-zA-Z0-9_-]+)\s*\((.*)\)?$", content, re.DOTALL)
            if fn_match:
                t_name = fn_match.group(1)
                args_str = fn_match.group(2).strip().rstrip(")")
                t_args = {}

                if not args_str:
                    calls.append((t_name, {}))
                    continue

                # Single string literal: name("query")
                single_str = re.match(r"^[\"'](.*)[\"']$", args_str)
                if single_str:
                    # Map to the first parameter of the tool if known
                    tool = self.tools.get(t_name)
                    param_name = tool.parameters[0]["name"] if tool and tool.parameters else "query"
                    t_args[param_name] = single_str.group(1)
                    calls.append((t_name, t_args))
                    continue

                # Key=value pairs: name(query="cormen", limit=5)
                kv_matches = re.findall(r"([a-zA-Z0-9_-]+)\s*=\s*(?:[\"']([^\"']*)[\"']|([^,\s\)]+))", args_str)
                if kv_matches:
                    for k, v1, v2 in kv_matches:
                        t_args[k] = v1 if v1 else v2
                    calls.append((t_name, t_args))
                    continue

                # Fallback: assign entire string to first parameter
                tool = self.tools.get(t_name)
                param_name = tool.parameters[0]["name"] if tool and tool.parameters else "query"
                t_args[param_name] = args_str.strip("\"'")
                calls.append((t_name, t_args))

        return calls

    def format_tools_for_prompt(self) -> str:
        """Render markdown summary of available tools for system prompt injection."""
        if not self.tools:
            return ""

        lines = ["Available tools:"]
        for tool in self.tools.values():
            params_desc = []
            for p in tool.parameters:
                req = "required" if p.get("required", False) else "optional"
                params_desc.append(f"{p.get('name')} ({p.get('type', 'string')}, {req}): {p.get('description', '')}")
            p_str = "; ".join(params_desc) if params_desc else "None"

            lines.append(f"- {tool.name}: {tool.description}")
            lines.append(f"  Parameters: {p_str}")

        lines.append("\nTo use a tool, reply in this format:")
        for t_name, tool in list(self.tools.items())[:2]:
            p_name = tool.parameters[0]["name"] if tool.parameters else "query"
            lines.append(f'<tool_call>{t_name}({p_name}="value")</tool_call>')
        return "\n".join(lines)


def execute_tool(tools: dict[str, Tool], tool_name: str, arguments: dict[str, Any], session_id: Optional[str] = None) -> str:
    executor = ToolExecutor(tools)
    return executor.execute(tool_name, arguments, session_id=session_id)
