from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from data_utils import normalize_transactions
from paths import DATABASE_PATH


TABLE_NAME = "transactions"


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    merchant TEXT NOT NULL,
    description TEXT NOT NULL,
    amount REAL NOT NULL,
    category TEXT NOT NULL,
    profile TEXT NOT NULL,
    bank TEXT NOT NULL,
    currency TEXT NOT NULL
);
"""


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_TABLE_SQL)
    return conn


def load_dataframe(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)


def save_transactions(df: pd.DataFrame, mode: str = "replace", db_path: Path = DATABASE_PATH) -> int:
    data = normalize_transactions(df)
    if mode not in {"replace", "append"}:
        raise ValueError("mode must be 'replace' or 'append'")

    with get_connection(db_path) as conn:
        if mode == "replace":
            conn.execute(f"DELETE FROM {TABLE_NAME}")
        data.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
    return len(data)


def validate_select_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    if not cleaned:
        raise ValueError("No SQL query was generated.")
    if not re.match(r"^select\b", cleaned, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")
    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "pragma", "attach", "detach"]
    lowered = cleaned.lower()
    if any(re.search(rf"\b{word}\b", lowered) for word in forbidden):
        raise ValueError("Unsafe SQL keyword detected.")
    return cleaned


def run_select_query(sql: str, db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    safe_sql = validate_select_sql(sql)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(safe_sql, conn)
