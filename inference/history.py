from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Optional


def init_history_db(db_path: Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON messages (session_id, id)"
        )
        conn.commit()


def save_turn(db_path: Path, session_id: str, role: str, content: str):
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content.strip(), datetime.now().isoformat()),
        )
        conn.commit()


def get_history(db_path: Path, session_id: str, limit: int = 8) -> list[dict[str, str]]:
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = cursor.fetchall()

    # Return in chronological order
    rows.reverse()
    return [{"role": r[0], "content": r[1]} for r in rows]


def get_all_sessions(db_path: Path) -> list[str]:
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT DISTINCT session_id FROM messages ORDER BY id DESC")
        return [row[0] for row in cursor.fetchall()]


def clear_history(db_path: Path, session_id: Optional[str] = None):
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        if session_id:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        else:
            conn.execute("DELETE FROM messages")
        conn.commit()
