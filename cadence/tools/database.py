"""Database tools — run SQL queries against SQLite databases."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from cadence.core.types import PermissionTier
from cadence.tools.base import Tool

# Statements that modify data or schema
_WRITE_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE"}

# Regex to detect write operations even when hidden behind CTEs, comments, or
# compound statements.  Strips SQL comments first, then checks all statement
# boundaries for write keywords.
_SQL_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


def _contains_write_operation(query: str) -> str | None:
    """Return the first write keyword found in *query*, or None if read-only."""
    # Strip comments so they can't hide write operations
    cleaned = _SQL_COMMENT_RE.sub(" ", query)
    # Split on semicolons to catch compound statements
    for stmt in cleaned.split(";"):
        tokens = stmt.strip().split()
        if not tokens:
            continue
        word = tokens[0].upper()
        if word in _WRITE_KEYWORDS:
            return word
        # Catch "WITH ... INSERT/UPDATE/DELETE" (CTE-based writes)
        upper = stmt.upper()
        if word == "WITH":
            for kw in ("INSERT", "UPDATE", "DELETE"):
                if kw in upper:
                    return kw
    return None


class SqlQueryTool(Tool):
    name = "sql_query"
    description = (
        "Execute a SQL query against a SQLite database. "
        "Read-only by default — set allow_write=true for INSERT/UPDATE/DELETE. "
        "Returns results as a formatted table."
    )
    parameters = {
        "type": "object",
        "properties": {
            "database": {
                "type": "string",
                "description": "Path to the SQLite database file. Use ':memory:' for a temporary database.",
            },
            "query": {
                "type": "string",
                "description": "SQL query to execute.",
            },
            "allow_write": {
                "type": "boolean",
                "description": "Allow write operations (INSERT, UPDATE, DELETE, etc.).",
                "default": False,
            },
            "max_rows": {
                "type": "integer",
                "description": "Maximum number of rows to return.",
                "default": 100,
            },
        },
        "required": ["database", "query"],
    }
    permission_tier = PermissionTier.STANDARD

    async def execute(
        self,
        database: str,
        query: str,
        allow_write: bool = False,
        max_rows: int = 100,
    ) -> str:
        # Safety check for write operations
        write_kw = _contains_write_operation(query)
        if write_kw and not allow_write:
            return f"Write operation '{write_kw}' blocked. Set allow_write=true to allow."

        if database != ":memory:":
            db_path = Path(database).expanduser()
            if not db_path.exists():
                return f"Database not found: {database}"

        conn = None
        try:
            conn = sqlite3.connect(database)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)

            if write_kw:
                conn.commit()
                return f"OK — {cursor.rowcount} rows affected."

            rows = cursor.fetchmany(max_rows + 1)
            if not rows:
                return "(no results)"

            # Format as table
            columns = rows[0].keys()
            header = " | ".join(columns)
            separator = "-+-".join("-" * len(c) for c in columns)
            lines = [header, separator]
            for row in rows[:max_rows]:
                lines.append(" | ".join(str(row[c]) for c in columns))

            if len(rows) > max_rows:
                lines.append(f"... ({max_rows} rows shown, more available)")

            return "\n".join(lines)

        except sqlite3.Error as e:
            return f"SQL error: {e}"
        finally:
            if conn:
                conn.close()
