import csv
import json
from pathlib import Path
from typing import Any, Optional

from tools.base import Tool


class FileTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        file_path: str,
        search_columns: Optional[list[str]] = None,
        parameters: Optional[list[dict[str, Any]]] = None,
        response_template: Optional[str] = None,
        base_dir: Optional[Path] = None,
        max_rows: int = 20,
    ):
        super().__init__(name, description, parameters, response_template)
        self.file_path_str = file_path
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.search_columns = search_columns or []
        self.max_rows = max_rows

    def _resolve_path(self) -> Path:
        p = Path(self.file_path_str)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def execute(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        target_path = self._resolve_path()
        if not target_path.exists():
            return [{"error": f"Data file not found: {target_path}"}]

        query = str(arguments.get("query", "")).strip().lower()

        rows: list[dict[str, Any]] = []
        ext = target_path.suffix.lower()

        if ext == ".csv":
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(dict(r))
        elif ext == ".json":
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    rows = [r for r in data if isinstance(r, dict)]
                elif isinstance(data, dict):
                    rows = [data]
        else:
            return [{"error": f"Unsupported file type: {ext}"}]

        if not query:
            return rows[:self.max_rows]

        matches = []
        for r in rows:
            cols = self.search_columns or list(r.keys())
            matched = False
            for col in cols:
                val = str(r.get(col, "")).lower()
                if query in val:
                    matched = True
                    break
            if matched:
                matches.append(r)
                if len(matches) >= self.max_rows:
                    break

        return matches
