import difflib
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Optional

import deps
from tools.base import Tool


def generate_spelling_variants(word: str) -> set[str]:
    """Generate standard English morphological variants (UK <-> US alternations)."""
    variants = set()
    w = word.lower()

    # 1. our <-> or (e.g. behaviour <-> behavior, colour <-> color, honour <-> honor)
    if "our" in w:
        variants.add(w.replace("our", "or"))
    elif not w.startswith("or") and not w.startswith("cor") and re.search(r"[a-z]or(s|ed|ing|al|able)?$", w) and len(w) >= 5:
        variants.add(re.sub(r"or((?:s|ed|ing|al|able)?)$", r"our\1", w))

    # 2. ise / isation <-> ize / ization (e.g. organisation <-> organization)
    if "isation" in w:
        variants.add(w.replace("isation", "ization"))
    elif "ization" in w:
        variants.add(w.replace("ization", "isation"))
    elif "ise" in w and not w.endswith("promise") and not w.endswith("exercise"):
        variants.add(w.replace("ise", "ize"))
    elif "ize" in w:
        variants.add(w.replace("ize", "ise"))

    # 3. ll <-> l before suffixes (modelling <-> modeling, cancelling <-> canceling)
    if re.search(r"ll(ing|ed|er)$", w):
        variants.add(re.sub(r"ll(ing|ed|er)$", r"l\1", w))
    elif re.search(r"l(ing|ed|er)$", w) and not re.search(r"[aeiou]{2}l", w):
        variants.add(re.sub(r"l(ing|ed|er)$", r"ll\1", w))

    # 4. re <-> er at word end (theatre <-> theater, centre <-> center, fibre <-> fiber)
    if w.endswith("re") and len(w) >= 5:
        variants.add(w[:-2] + "er")
    elif w.endswith("er") and len(w) >= 5:
        variants.add(w[:-2] + "re")

    # 5. ogue <-> og (dialogue <-> dialog, catalogue <-> catalog)
    if w.endswith("ogue"):
        variants.add(w[:-4] + "og")
    elif w.endswith("og") and len(w) >= 5:
        variants.add(w[:-2] + "ogue")

    # 6. amme <-> am (programme <-> program)
    if w.endswith("amme"):
        variants.add(w[:-4] + "am")
    elif w.endswith("am") and len(w) >= 6:
        variants.add(w + "me")

    variants.discard(w)
    return variants


def clean_conversational_phrases(text: str) -> str:
    """Strip conversational wrappers so queries match underlying database records directly."""
    # Protect domain technical terms like c++ or c# before stripping punctuation
    pre_cleaned = re.sub(r"\bc\+\+", "__CPP__", text.strip(), flags=re.IGNORECASE)
    pre_cleaned = re.sub(r"\bc#\b", "__CSHARP__", pre_cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^\w\s]", " ", pre_cleaned)
    cleaned = cleaned.replace("__CPP__", "c++").replace("__CSHARP__", "c#")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Strip leading conversational requests
    cleaned = re.sub(
        r"^(could|can|would)\s+(?:i|we|you)?\s*(?:please\s+)?(?:have|get|tell\s+me\s+where|tell\s+me\s+about|tell\s+me|check\s+if|check|find|show\s+me|locate)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"^(please\s+)?(?:tell\s+me\s+where|tell\s+me\s+about|tell\s+me|show\s+me|find|check\s+if|check|search\s+for|search|locate|give\s+me)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"^(do\s+(?:you|u)\s+have(?:\s+any)?|do\s+(?:you|u)\s+got|have\s+(?:you|u)\s+got|where\s+(?:can\s+(?:i|we)\s+find|is|are|to\s+find)|is\s+there|are\s+there|is|are|looking\s+for|i\s+want|i\s+am\s+looking\s+for)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"^(if|whether|the|a|an)\s+", "", cleaned, flags=re.IGNORECASE).strip()

    # Strip trailing conversational modifiers
    cleaned = re.sub(
        r"\s+(is|are)?\s*(available|present|kept|located|shelved|found)\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"\s+(available|there|present|here)(\s+right\s+now)?$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"\s+(right\s+now|currently|today)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"\s+(is|are)$", "", cleaned, flags=re.IGNORECASE).strip()

    return re.sub(r"\s+", " ", cleaned).strip()


def words_match(w1: str, w2: str) -> bool:
    """Check if two words match exactly, via English morphological variants, or substring/typo."""
    if w1 == w2:
        return True
    if w2 in generate_spelling_variants(w1) or w1 in generate_spelling_variants(w2):
        return True
    if len(w1) >= 4 and (w1 in w2 or w2 in w1):
        return True
    if difflib.SequenceMatcher(None, w1, w2).ratio() >= 0.80:
        return True
    return False


def escape_like_wildcards(val: str) -> str:
    """Escape % and _ for safe SQL LIKE matching unless intended."""
    return val.replace("%", "\\%").replace("_", "\\_")



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
            if isinstance(val, str) and val.strip():
                cleaned = clean_conversational_phrases(val)
                if cleaned:
                    val = cleaned
            values.append(val if val is not None else "")

        db_type = self.connection.get("type", "sqlite").lower()
        if db_type == "sqlite":
            return self._execute_sqlite(self.query, arguments, values)
        elif db_type == "mysql":
            return self._execute_mysql(self.query, arguments, values)
        else:
            raise NotImplementedError(f"Database type '{db_type}' is not currently supported.")
