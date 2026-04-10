from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from face_service import FaceService


class DetectResponse(BaseModel):
    faces: List[dict]
    image_size: dict
    storage: dict


class SimilarResponse(BaseModel):
    matches: List[dict]


service = FaceService()
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Face Detection and Similarity API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    """Serve the front-end page if it exists."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Static UI not found. Upload files to /faces/detect or /faces/similar."}


@app.post("/faces/detect", response_model=DetectResponse)
async def detect_faces(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty")

    analysis = service.analyze_image(data)
    embeddings = analysis.get("embeddings", [])
    # Automatically persist every detection request to support later similarity lookups.
    storage = service.store_embeddings(file.filename, data, embeddings)
    return DetectResponse(
        faces=analysis.get("faces", []),
        image_size=analysis.get("image_size", {"width": 0, "height": 0}),
        storage=storage,
    )


@app.post("/faces/store")
async def store_face(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty")

    analysis = service.analyze_image(data)
    embeddings = analysis.get("embeddings", [])
    result = service.store_embeddings(file.filename, data, embeddings)
    return {"result": result}


@app.post("/faces/similar", response_model=SimilarResponse)
async def search_similar(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=50, description="Number of nearest matches to return"),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty")

    analysis = service.analyze_image(data)
    embeddings = analysis.get("embeddings", [])
    matches = service.search_similar(embeddings, top_k)
    return SimilarResponse(matches=matches)
