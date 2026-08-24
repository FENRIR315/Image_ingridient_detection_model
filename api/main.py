import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from ultralytics import YOLO

MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "best.pt")
IMGSZ = 640

model = YOLO(MODEL_PATH)

app = FastAPI(title="Ingredient Detector API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file")
    return img


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Query(0.30, gt=0.0, lt=1.0)):
    data = await file.read()
    img = decode(data)
    t0 = time.time()
    res = model.predict(img, imgsz=IMGSZ, conf=conf, device=0, verbose=False)[0]
    ms = round((time.time() - t0) * 1000)

    detections = [
        {
            "name": model.names[int(c)],
            "confidence": round(float(s), 3),
            "box": [round(v) for v in box.tolist()],
        }
        for c, s, box in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy)
    ]
    detections.sort(key=lambda d: -d["confidence"])

    best: dict[str, dict] = {}
    for det in detections:
        best.setdefault(det["name"], det)
    ingredients = [{"name": d["name"], "confidence": d["confidence"]} for d in best.values()]

    return {"count": len(ingredients), "inference_ms": ms, "ingredients": ingredients, "detections": detections}


@app.post("/detect/annotated")
async def detect_annotated(file: UploadFile = File(...), conf: float = Query(0.30, gt=0.0, lt=1.0)):
    data = await file.read()
    img = decode(data)
    res = model.predict(img, imgsz=IMGSZ, conf=conf, device=0, verbose=False)[0]
    ok, jpg = cv2.imencode(".jpg", res.plot(), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image")
    return Response(jpg.tobytes(), media_type="image/jpeg")
