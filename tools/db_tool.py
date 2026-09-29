import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Optional

import deps
from tools.base import Tool


class DbTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        query: str,
        connection: dict[str, Any],
        parameters: Optional[list[dict[str, Any]]] = None,
        response_template: Optional[str] = None,
        max_rows: int = 20,
    ):
        super().__init__(name, description, parameters, response_template)
        self.query = query.strip()
        self.connection = connection
        self.max_rows = max_rows
        self._validate_query_safety(self.query)

    @staticmethod
    def _validate_query_safety(query: str):
        """Only allow SELECT or WITH at the start. No multi-statement tricks."""
        normalized = query.strip().lower()
        if not (normalized.startswith("select") or normalized.startswith("with")):
            raise ValueError("Only SELECT or WITH queries are permitted in DbTool.")

        # Block multiple statements (semicolons outside of string literals)
        # Simple check: strip everything inside single quotes, then look for ;
        stripped = re.sub(r"'[^']*'", "", normalized)
        if ";" in stripped:
            raise ValueError("Multiple SQL statements are not allowed in DbTool.")

    def _expand_values(self, placeholder_count: int, values: list[Any]) -> list[Any]:
        """Repeat values to match the number of placeholders in the query.

        If a query has 4 placeholders and 2 values, each value gets used twice:
        [a, b] -> [a, b, a, b]. If there is 1 value and 3 placeholders: [a] -> [a, a, a].
        """
        if not values or placeholder_count == 0:
            return values
        if len(values) == placeholder_count:
            return values
        if placeholder_count % len(values) == 0:
            return values * (placeholder_count // len(values))
        # Fallback: repeat the full list enough times and truncate
        repeats = (placeholder_count // len(values)) + 1
        return (values * repeats)[:placeholder_count]

    def _execute_sqlite(self, query: str, arguments: dict[str, Any], values: list[Any]) -> list[dict[str, Any]]:
        db_path = self.connection.get("path")
        if not db_path:
            raise ValueError("SQLite connection requires 'path' setting.")

        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        # Adapt %s to ? for SQLite
        adapted_query = query.replace("%s", "?")

        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if re.search(r":\w+", adapted_query):
                cursor.execute(adapted_query, arguments)
            else:
                placeholder_count = adapted_query.count("?")
                actual_values = self._expand_values(placeholder_count, values)
                cursor.execute(adapted_query, actual_values)
            rows = cursor.fetchmany(self.max_rows)
            return [dict(row) for row in rows]

    def _execute_mysql(self, query: str, arguments: dict[str, Any], values: list[Any]) -> list[dict[str, Any]]:
        if not deps.require_group("db"):
            raise RuntimeError("Database dependencies are required for MySQL connection.")

        import MySQLdb
        import MySQLdb.cursors

        password = self.connection.get("password")
        password_env = self.connection.get("password_env")
        if password_env and not password:
            password = os.environ.get(password_env, "")

        # Adapt ? to %s for MySQL
        adapted_query = query.replace("?", "%s")

        conn = MySQLdb.connect(
            host=self.connection.get("host", "localhost"),
            port=int(self.connection.get("port", 3306)),
            user=self.connection.get("user", "root"),
            passwd=password or "",
            db=self.connection.get("database", ""),
            cursorclass=MySQLdb.cursors.DictCursor,
        )
        try:
            with conn.cursor() as cursor:
                placeholder_count = adapted_query.count("%s")
                actual_values = self._expand_values(placeholder_count, values)
                cursor.execute(adapted_query, actual_values)
                rows = cursor.fetchmany(self.max_rows)
                return list(rows)
        finally:
            conn.close()

    def execute(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        # Map parameter values in declared order
        values = []
        for p in self.parameters:
            val = arguments.get(p["name"])
            values.append(val if val is not None else "")

        db_type = self.connection.get("type", "sqlite").lower()
        if db_type == "sqlite":
            return self._execute_sqlite(self.query, arguments, values)
        elif db_type == "mysql":
            return self._execute_mysql(self.query, arguments, values)
        else:
            raise NotImplementedError(f"Database type '{db_type}' is not currently supported.")
