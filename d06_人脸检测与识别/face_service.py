import base64
import io
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image


class FaceService:
    """Encapsulates face detection, storage, and similarity utilities."""

    def __init__(self, storage_dir: str = "stored_images", db_path: str = "face_data.db") -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self._lock = threading.Lock()

        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=True,
            device=self.device,
        )
        self.resnet = self._load_resnet()
        self._init_db()

    def _load_resnet(self) -> InceptionResnetV1:
        """Load the embedding model, preferring the cached checkpoint if available."""
        checkpoint = Path.home() / ".cache/torch/hub/checkpoints/20180402-114759-vggface2.pt"
        if checkpoint.exists():
            resnet = InceptionResnetV1(pretrained=None)
            state_dict = torch.load(checkpoint, map_location=self.device)
            resnet.load_state_dict(state_dict, strict=False)
        else:
            resnet = InceptionResnetV1(pretrained="vggface2")
        return resnet.eval().to(self.device)

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE,
                    path TEXT NOT NULL,
                    embeddings TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run detection plus embedding extraction for one uploaded image."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        boxes, probs = self.mtcnn.detect(image)
        aligned_faces, _ = self.mtcnn(image, return_prob=True)

        faces_payload: List[Dict[str, Any]] = []
        if boxes is not None and probs is not None:
            for box, score in zip(boxes, probs):
                if box is None or score is None:
                    continue
                crop = image.crop(tuple(box))
                buffer = io.BytesIO()
                crop.save(buffer, format="PNG")
                faces_payload.append(
                    {
                        "box": {
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                        },
                        "probability": float(score),
                        "thumbnail": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                    }
                )

        face_tensors: List[torch.Tensor] = []
        if aligned_faces is not None:
            if isinstance(aligned_faces, torch.Tensor):
                face_tensors = [aligned_faces[i] for i in range(aligned_faces.shape[0])]
            else:
                face_tensors = list(aligned_faces)

        embeddings = self._compute_embeddings(face_tensors)
        return {
            "faces": faces_payload,
            "embeddings": embeddings,
            "image_size": {"width": width, "height": height},
        }

    def _compute_embeddings(self, face_tensors: List[torch.Tensor]) -> List[List[float]]:
        if not face_tensors:
            return []
        batch = torch.stack(face_tensors).to(self.device)
        with torch.no_grad():
            vectors = self.resnet(batch).cpu()
        return [vec.tolist() for vec in vectors]

    def store_embeddings(self, filename: str, image_bytes: bytes, embeddings: List[List[float]]) -> Dict[str, Any]:
        if not embeddings:
            return {"stored": False, "reason": "no_faces"}

        safe_name = Path(filename).name or f"upload_{uuid.uuid4().hex}.png"
        destination = self.storage_dir / safe_name

        with self._lock, sqlite3.connect(self.db_path) as conn:
            if conn.execute("SELECT 1 FROM faces WHERE filename = ?", (safe_name,)).fetchone():
                return {"stored": False, "reason": "duplicate", "filename": safe_name}

            destination.write_bytes(image_bytes)
            conn.execute(
                "INSERT INTO faces (filename, path, embeddings) VALUES (?, ?, ?)",
                (safe_name, str(destination), json.dumps(embeddings)),
            )

        return {"stored": True, "filename": safe_name}

    def search_similar(self, embeddings: List[List[float]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not embeddings:
            return []

        query = np.array(embeddings, dtype=np.float32)
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT filename, path, embeddings FROM faces").fetchall()

        matches: List[Dict[str, Any]] = []
        if not rows:
            return matches

        for filename, path, stored_json in rows:
            stored_vectors = json.loads(stored_json)
            if not stored_vectors:
                continue
            stored = np.array(stored_vectors, dtype=np.float32)
            distances = self._pairwise_distances(query, stored)
            best_distance = float(np.min(distances))
            matches.append(
                {
                    "filename": filename,
                    "distance": best_distance,
                    "image": self._read_image_base64(path),
                }
            )

        matches.sort(key=lambda item: item["distance"])
        return matches[: max(0, top_k)]

    @staticmethod
    def _pairwise_distances(query: np.ndarray, stored: np.ndarray) -> np.ndarray:
        expanded_query = query[:, None, :]
        expanded_stored = stored[None, :, :]
        return np.linalg.norm(expanded_query - expanded_stored, axis=2)

    @staticmethod
    def _read_image_base64(path: str) -> str:
        try:
            data = Path(path).read_bytes()
            return base64.b64encode(data).decode("utf-8")
        except FileNotFoundError:
            return ""
