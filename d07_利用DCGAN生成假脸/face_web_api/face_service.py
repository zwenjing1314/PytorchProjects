import base64
import io
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageDraw


class FaceService:
    """封装人脸检测、特征提取和相似度计算。"""

    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=True,
        )
        self.resnet = self._load_resnet()

    def _load_resnet(self) -> InceptionResnetV1:
        weights_path = os.getenv("FACE_RECOG_WEIGHTS")
        model = InceptionResnetV1(pretrained=None)

        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
        else:
            # 如果没有配置本地权重，则使用 facenet_pytorch 的预训练权重。
            model = InceptionResnetV1(pretrained="vggface2")

        return model.eval().to(self.device)

    @staticmethod
    def load_image(image_bytes: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image

    @staticmethod
    def image_to_base64(image: Image.Image, image_format: str = "JPEG") -> str:
        buffer = io.BytesIO()
        image.save(buffer, format=image_format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{image_format.lower()};base64,{encoded}"

    def detect_faces(self, image: Image.Image) -> dict[str, Any]:
        boxes, probs = self.mtcnn.detect(image)
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        detections: list[dict[str, Any]] = []
        cropped_faces: list[Image.Image] = []

        if boxes is None or probs is None:
            return {
                "face_count": 0,
                "detections": [],
                "annotated_image": self.image_to_base64(annotated),
                "cropped_faces": [],
            }

        width, height = image.size
        for idx, (box, prob) in enumerate(zip(boxes, probs, strict=False), start=1):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 16)), f"{prob:.4f}", fill="red")

            face_image = image.crop((x1, y1, x2, y2))
            cropped_faces.append(face_image)
            detections.append(
                {
                    "index": idx,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(float(prob), 6),
                }
            )

        return {
            "face_count": len(detections),
            "detections": detections,
            "annotated_image": self.image_to_base64(annotated),
            "cropped_faces": [self.image_to_base64(face) for face in cropped_faces],
        }

    def get_primary_embedding(self, image: Image.Image) -> tuple[list[float], dict[str, Any]]:
        boxes, probs = self.mtcnn.detect(image)
        aligned_faces = self.mtcnn(image)

        if boxes is None or probs is None or aligned_faces is None:
            raise ValueError("当前图片中未检测到可用于比对的人脸。")

        # 如果图片里有多张人脸，默认使用置信度最高的人脸做入库和检索。
        if aligned_faces.ndim == 3:
            aligned_faces = aligned_faces.unsqueeze(0)

        best_index = int(np.argmax(probs))
        best_face = aligned_faces[best_index].unsqueeze(0).to(self.device)
        embedding = self.resnet(best_face).detach().cpu().numpy()[0].astype(float)

        detection_result = self.detect_faces(image)
        detection_result["primary_face_index"] = best_index + 1
        detection_result["primary_confidence"] = round(float(probs[best_index]), 6)

        return embedding.tolist(), detection_result

    @staticmethod
    def compute_top_k(
        query_embedding: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
        exclude_filename: str | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = np.asarray(query_embedding, dtype=np.float32)
        scored: list[dict[str, Any]] = []

        for item in candidates:
            if exclude_filename and item["filename"] == exclude_filename:
                continue

            candidate_vector = np.asarray(item["embedding"], dtype=np.float32)
            distance = float(np.linalg.norm(query_vector - candidate_vector))
            score = 1.0 / (1.0 + distance)
            scored.append(
                {
                    "id": item["id"],
                    "filename": item["filename"],
                    "stored_path": item["stored_path"],
                    "distance": round(distance, 6),
                    "similarity_score": round(score, 6),
                    "face_count": item["face_count"],
                    "face_confidence": item["face_confidence"],
                    "created_at": item["created_at"],
                }
            )

        scored.sort(key=lambda x: x["distance"])
        return scored[:top_k]

    @staticmethod
    def build_storage_name(filename: str) -> str:
        ext = Path(filename).suffix.lower() or ".jpg"
        unique_id = uuid4().hex
        return f"{unique_id}{ext}"
