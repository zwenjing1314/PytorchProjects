from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from face_web_api.database import FaceImageRepository
from face_web_api.face_service import FaceService


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "runtime_data"
UPLOAD_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "face_images.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Face Detection and Retrieval API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploaded", StaticFiles(directory=UPLOAD_DIR), name="uploaded")

repository = FaceImageRepository(DATABASE_PATH)
face_service = FaceService()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/detect")
async def detect_faces(file: UploadFile = File(...)) -> dict:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空。")

    try:
        image = face_service.load_image(image_bytes)
        return face_service.detect_faces(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/search")
async def search_similar_faces(
    file: UploadFile = File(...),
    top_k: int = Form(5),
) -> dict:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空。")

    top_k = max(1, min(top_k, 20))

    try:
        image = face_service.load_image(image_bytes)
        embedding, detection_result = face_service.get_primary_embedding(image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"处理图片失败: {exc}") from exc

    filename = file.filename or "uploaded_image.jpg"
    existing = repository.get_by_filename(filename)

    if existing is None:
        stored_name = face_service.build_storage_name(filename)
        stored_path = UPLOAD_DIR / stored_name
        stored_path.write_bytes(image_bytes)
        repository.insert_image(
            filename=filename,
            stored_path=f"/uploaded/{stored_name}",
            embedding=embedding,
            face_count=detection_result["face_count"],
            face_confidence=detection_result["primary_confidence"],
        )
        duplicate_skipped = False
    else:
        duplicate_skipped = True

    matches = face_service.compute_top_k(
        query_embedding=embedding,
        candidates=repository.list_all_embeddings(),
        top_k=top_k,
        exclude_filename=filename,
    )

    return {
        "duplicate_skipped": duplicate_skipped,
        "query_filename": filename,
        "top_k": top_k,
        "detection": detection_result,
        "matches": matches,
    }
