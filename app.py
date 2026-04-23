import cv2
import numpy as np
import face_recognition
import os
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Smart AI Attendance System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.json"

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

known_face_encodings = []
known_face_names = []

# ---------------- LOAD FACES ----------------
def load_registered_faces():
    global known_face_encodings, known_face_names
    new_encodings = []
    new_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    new_encodings.append(encodings[0])
                    new_names.append(os.path.splitext(filename)[0])
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    known_face_encodings = new_encodings
    known_face_names = new_names
    print(f"--- Model Sync Complete: {len(known_face_names)} faces active ---")

load_registered_faces()

# ---------------- ATTENDANCE LOGIC ----------------
def log_attendance(name: str):
    if name == "Unknown":
        return {"status": "ignored", "message": ""}

    records = []
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, "r") as f:
                records = json.load(f)
        except:
            records = []

    # ✅ FIXED: hour format with 'T'
    current_hour = datetime.now().strftime("%Y-%m-%dT%H")

    already_logged = any(
        r['name'] == name and r['timestamp'].startswith(current_hour)
        for r in records
    )

    if already_logged:
        return {
            "status": "exists",
            "message": f"Hey {name}, your attendance is already marked"
        }

    # ✅ Mark attendance automatically
    new_record = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "status": "Present"
    }

    records.append(new_record)

    with open(ATTENDANCE_FILE, "w") as f:
        json.dump(records, f, indent=2)

    return {
        "status": "marked",
        "message": f"Hey {name}, your attendance is marked"
    }

# ---------------- REGISTER ----------------
@app.post("/register/{name}")
async def register_face(name: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_')]).strip()
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid name")

        file_path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}.jpg")

        contents = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        background_tasks.add_task(load_registered_faces)

        return {
            "status": "success",
            "message": f"Identity '{safe_name}' captured. Processing in background."
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ---------------- GET ATTENDANCE ----------------
@app.get("/attendance")
async def get_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    with open(ATTENDANCE_FILE, "r") as f:
        try:
            return json.load(f)
        except:
            return []

# ---------------- RECOGNITION ----------------
@app.post("/recognize")
async def recognize_api(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"detections": []}

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detections = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        confidence = 0.0
        message = ""

        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = 1.0 - face_distances[best_match_index]

        # ✅ AUTO attendance + message
        if name != "Unknown":
            response = log_attendance(name)
            message = response["message"]

        detections.append({
            "name": name,
            "box": [int(top), int(right), int(bottom), int(left)],
            "confidence": float(confidence),
            "message": message
        })

    return {"detections": detections}

# ---------------- RUN ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)