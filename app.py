import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json

app = FastAPI(title="Smart AI Attendance System")
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.json"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

known_face_encodings = []
known_face_names = []

def load_registered_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])
    print(f"Loaded {len(known_face_names)} registered faces.")

load_registered_faces()

class AttendanceRecord(BaseModel):
    name: str
    timestamp: str
    status: str


def process_frame(frame):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        
        results.append({
            "name": name,
            "box": [top, right, bottom, left],
            "confidence": float(1 - face_distances[best_match_index]) if len(face_distances) > 0 else 0.0
        })
    return results
# log attendance if recognized and not already logged for today samjhe 
def log_attendance(name: str):

    if name == "Unknown":
        return
        
    records = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            records = json.load(f)
    
    
    today = datetime.now().strftime("%Y-%m-%d")
    for r in records:
        if r['name'] == name and r['timestamp'].startswith(today):
            return 

    new_record = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "status": "Present"
    }
    records.append(new_record)
    with open(ATTENDANCE_FILE, "w") as f:
        json.dump(records, f)


@app.post("/register/{name}")
async def register_face(name: str, file: UploadFile = File(...)):
    
    file_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    
    load_registered_faces()
    return {"message": f"User {name} registered successfully."}

@app.get("/attendance")
async def get_attendance():
    
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    with open(ATTENDANCE_FILE, "r") as f:
        return json.load(f)

@app.post("/recognize")
async def recognize_api(file: UploadFile = File(...)):
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    identities = process_frame(img)
    for person in identities:
        log_attendance(person['name'])
        
    return {"detections": identities}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
