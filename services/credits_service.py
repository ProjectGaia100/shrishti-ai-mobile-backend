"""
Credits service
Persistent per-user credits for prediction usage.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Dict, List, Optional

DEFAULT_CREDITS = 30

BUNDLES: List[Dict[str, object]] = [
    {"id": "plus_1000", "credits": 1000, "price_inr": 999, "label": "1000 Credits"},
    {"id": "plus_10000", "credits": 10000, "price_inr": 7999, "label": "10000 Credits"},
    {"id": "plus_100000", "credits": 100000, "price_inr": 59999, "label": "100000 Credits"},
]


class CreditsService:
    def __init__(self, db_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        default_db = os.path.join(base_dir, "proj_data", "credits.sqlite3")
        self.db_path = db_path or default_db
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_credits (
                    user_id TEXT PRIMARY KEY,
                    credits INTEGER NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS credit_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    tx_type TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    balance_after INTEGER NOT NULL,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def _ensure_user(self, conn: sqlite3.Connection, user_id: str) -> None:
        row = conn.execute(
            "SELECT user_id FROM user_credits WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO user_credits (user_id, credits) VALUES (?, ?)",
                (user_id, DEFAULT_CREDITS),
            )
            conn.execute(
                """
                INSERT INTO credit_transactions (user_id, tx_type, amount, balance_after, description)
                VALUES (?, 'grant', ?, ?, ?)
                """,
                (user_id, DEFAULT_CREDITS, DEFAULT_CREDITS, "Initial credits"),
            )

    def get_balance(self, user_id: str) -> int:
        with self._lock:
            with self._connect() as conn:
                self._ensure_user(conn, user_id)
                row = conn.execute(
                    "SELECT credits FROM user_credits WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                conn.commit()
                return int(row["credits"]) if row else DEFAULT_CREDITS

    def charge(self, user_id: str, amount: int, description: str) -> Dict[str, object]:
        if amount <= 0:
            return {"success": True, "remaining_credits": self.get_balance(user_id)}

        with self._lock:
            with self._connect() as conn:
                self._ensure_user(conn, user_id)
                row = conn.execute(
                    "SELECT credits FROM user_credits WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                current_credits = int(row["credits"]) if row else DEFAULT_CREDITS

                if current_credits < amount:
                    conn.commit()
                    return {
                        "success": False,
                        "error": "INSUFFICIENT_CREDITS",
                        "remaining_credits": current_credits,
                        "required_credits": amount,
                    }

                new_balance = current_credits - amount
                conn.execute(
                    "UPDATE user_credits SET credits = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (new_balance, user_id),
                )
                conn.execute(
                    """
                    INSERT INTO credit_transactions (user_id, tx_type, amount, balance_after, description)
                    VALUES (?, 'charge', ?, ?, ?)
                    """,
                    (user_id, -amount, new_balance, description),
                )
                conn.commit()

                return {
                    "success": True,
                    "remaining_credits": new_balance,
                    "charged_credits": amount,
                }

    def purchase_bundle(self, user_id: str, bundle_id: str) -> Dict[str, object]:
        bundle = next((b for b in BUNDLES if b["id"] == bundle_id), None)
        if bundle is None:
            return {"success": False, "error": "Invalid bundle selected"}

        add_credits = int(bundle["credits"])

        with self._lock:
            with self._connect() as conn:
                self._ensure_user(conn, user_id)
                row = conn.execute(
                    "SELECT credits FROM user_credits WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                current_credits = int(row["credits"]) if row else DEFAULT_CREDITS
                new_balance = current_credits + add_credits

                conn.execute(
                    "UPDATE user_credits SET credits = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (new_balance, user_id),
                )
                conn.execute(
                    """
                    INSERT INTO credit_transactions (user_id, tx_type, amount, balance_after, description)
                    VALUES (?, 'purchase', ?, ?, ?)
                    """,
                    (user_id, add_credits, new_balance, f"Bundle: {bundle['label']}"),
                )
                conn.commit()

                return {
                    "success": True,
                    "remaining_credits": new_balance,
                    "bundle": bundle,
                }

    def get_bundles(self) -> List[Dict[str, object]]:
        return BUNDLES


_credit_service: Optional[CreditsService] = None


def get_credit_service() -> CreditsService:
    global _credit_service
    if _credit_service is None:
        _credit_service = CreditsService()
    return _credit_service
