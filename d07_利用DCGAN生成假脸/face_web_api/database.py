import json
import sqlite3
from pathlib import Path
from typing import Any


class FaceImageRepository:
    """负责管理图片元数据和人脸特征向量的 SQLite 仓库。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS face_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    stored_path TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    face_count INTEGER NOT NULL,
                    face_confidence REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get_by_filename(self, filename: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM face_images WHERE filename = ?",
                (filename,),
            ).fetchone()
        return dict(row) if row else None

    def insert_image(
        self,
        filename: str,
        stored_path: str,
        embedding: list[float],
        face_count: int,
        face_confidence: float,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO face_images (
                    filename, stored_path, embedding_json, face_count, face_confidence
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    stored_path,
                    json.dumps(embedding, ensure_ascii=True),
                    face_count,
                    face_confidence,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_all_embeddings(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, stored_path, embedding_json, face_count, face_confidence, created_at
                FROM face_images
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["embedding"] = json.loads(item.pop("embedding_json"))
            result.append(item)
        return result
