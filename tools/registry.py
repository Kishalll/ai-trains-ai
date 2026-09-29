from pathlib import Path
from typing import Optional
import yaml

from tools.api_tool import ApiTool
from tools.base import Tool
from tools.db_tool import DbTool
from tools.file_tool import FileTool
from tools.static_tool import StaticTool


def load_tools(tools_file_path: Path | str, role_dir: Optional[Path | str] = None) -> dict[str, Tool]:
    file_path = Path(tools_file_path)
    if not file_path.exists():
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    global_conn = data.get("connection", {})
    tools_list = data.get("tools", [])
    registry: dict[str, Tool] = {}

    base_path = Path(role_dir) if role_dir else file_path.parent

    for item in tools_list:
        name = item.get("name")
        if not name:
            continue

        tool_type = item.get("type", "static").lower()
        desc = item.get("description", "")
        params = item.get("parameters", [])
        template = item.get("response_template")

        if tool_type == "db":
            conn = dict(item.get("connection") or global_conn)
            # Resolve relative SQLite paths against the role directory
            if conn.get("type", "sqlite").lower() == "sqlite" and conn.get("path"):
                db_path = Path(conn["path"])
                if not db_path.is_absolute():
                    conn["path"] = str(base_path / db_path)
            query = item.get("query", "")
            registry[name] = DbTool(
                name=name,
                description=desc,
                query=query,
                connection=conn,
                parameters=params,
                response_template=template,
                max_rows=item.get("max_rows", 20),
            )
        elif tool_type == "api":
            url = item.get("url", "")
            method = item.get("method", "GET")
            headers = item.get("headers")
            registry[name] = ApiTool(
                name=name,
                description=desc,
                url=url,
                method=method,
                parameters=params,
                headers=headers,
                response_template=template,
                timeout=float(item.get("timeout", 5.0)),
            )
        elif tool_type == "file":
            f_path = item.get("file_path", "")
            search_cols = item.get("search_columns", [])
            registry[name] = FileTool(
                name=name,
                description=desc,
                file_path=f_path,
                search_columns=search_cols,
                parameters=params,
                response_template=template,
                base_dir=base_path,
                max_rows=item.get("max_rows", 20),
            )
        elif tool_type == "static":
            response = item.get("response", "")
            registry[name] = StaticTool(
                name=name,
                description=desc,
                response=response,
                parameters=params,
                response_template=template,
            )

    return registry
