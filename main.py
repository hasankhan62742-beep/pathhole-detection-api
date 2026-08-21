from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
from datetime import datetime
import uuid

app = FastAPI(title="Pothole Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('best.pt')
all_detections = []

def classify_severity(box_area, confidence):
    if box_area > 15000 and confidence > 0.6:
        return "Severe"
    elif box_area > 7000:
        return "Moderate"
    else:
        return "Minor"

@app.post("/detect")
async def detect_pothole(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(source=img, conf=0.4, verbose=False)

    detections_in_frame = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)
            severity = classify_severity(area, conf)

            detection = {
                "id": str(uuid.uuid4())[:8],
                "latitude": latitude,
                "longitude": longitude,
                "confidence": round(conf, 2),
                "severity": severity,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)]
            }
            detections_in_frame.append(detection)
            all_detections.append(detection)

    return {
        "potholes_detected": len(detections_in_frame),
        "detections": detections_in_frame
    }

@app.get("/all-detections")
async def get_all_detections():
    return {"total": len(all_detections), "data": all_detections}

@app.get("/")
async def root():
    return {"status": "Pothole Detection API is live 🚀"}
