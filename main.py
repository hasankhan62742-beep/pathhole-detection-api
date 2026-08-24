from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
from datetime import datetime
from geopy.distance import geodesic
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
pending_candidates = []

CONFIRMATION_THRESHOLD = 2
LOCATION_RADIUS_M = 12

def classify_severity(box_area, confidence):
    if box_area > 15000 and confidence > 0.6:
        return "Severe"
    elif box_area > 7000:
        return "Moderate"
    else:
        return "Minor"

def find_nearby_candidate(lat, lon):
    for c in pending_candidates:
        if geodesic((lat, lon), (c["latitude"], c["longitude"])).meters < LOCATION_RADIUS_M:
            return c
    return None

def already_confirmed(lat, lon):
    for d in all_detections:
        if geodesic((lat, lon), (d["latitude"], d["longitude"])).meters < LOCATION_RADIUS_M:
            return True
    return False

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

    confirmed_this_frame = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)
            severity = classify_severity(area, conf)

            if already_confirmed(latitude, longitude):
                continue

            candidate = find_nearby_candidate(latitude, longitude)
            if candidate:
                candidate["count"] += 1
                candidate["confidence"] = max(candidate["confidence"], conf)
                candidate["severity"] = severity
            else:
                candidate = {
                    "latitude": latitude, "longitude": longitude,
                    "count": 1, "confidence": conf, "severity": severity
                }
                pending_candidates.append(candidate)

            if candidate["count"] >= CONFIRMATION_THRESHOLD:
                confirmed = {
                    "id": str(uuid.uuid4())[:8],
                    "latitude": latitude,
                    "longitude": longitude,
                    "confidence": round(candidate["confidence"], 2),
                    "severity": candidate["severity"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)]
                }
                all_detections.append(confirmed)
                confirmed_this_frame.append(confirmed)
                pending_candidates.remove(candidate)

    return {
        "confirmed_potholes": len(confirmed_this_frame),
        "detections": confirmed_this_frame
    }

@app.get("/all-detections")
async def get_all_detections():
    return {"total": len(all_detections), "data": all_detections}

@app.get("/")
async def root():
    return {"status": "Pothole Detection API v3 (temporal consistency enabled) is live 🚀"}
