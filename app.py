import cv2
import numpy as np
import face_recognition
import os
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- Configuration & State ---
app = FastAPI(title="Smart AI Attendance System")

# CRITICAL: Add CORS Middleware to allow the browser to talk to the backend
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

# In-memory cache for registered faces
known_face_encodings = []
known_face_names = []

def load_registered_faces():
    """Loads encodings from the known_faces directory. Optimized for speed."""
    global known_face_encodings, known_face_names
    new_encodings = []
    new_names = []
    
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                # We reload images to ensure the cache is always fresh
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

# Initial load
load_registered_faces()

# --- AI Logic ---
def log_attendance(name: str):
    if name == "Unknown":
        return
        
    records = []
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, "r") as f:
                records = json.load(f)
        except:
            records = []
    
    # Check for existing log in the current hour to avoid spamming
    current_hour = datetime.now().strftime("%Y-%m-%d %H")
    already_logged = any(
        r['name'] == name and r['timestamp'].startswith(current_hour) 
        for r in records
    )

    if not already_logged:
        new_record = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "status": "Present"
        }
        records.append(new_record)
        with open(ATTENDANCE_FILE, "w") as f:
            json.dump(records, f)

# --- API Endpoints ---
@app.post("/register/{name}")
async def register_face(name: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Registers a face. Uses BackgroundTasks to prevent browser timeouts 
    during heavy AI encoding processing.
    """
    try:
        # Clean name for file system
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_')]).strip()
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid name")

        file_path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}.jpg")
        
        contents = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        # We trigger the reload in the background so we can respond to the browser immediately
        background_tasks.add_task(load_registered_faces)
        
        return {"status": "success", "message": f"Identity '{safe_name}' captured. System is processing in background."}
    except Exception as e:
        print(f"Registration Error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/attendance")
async def get_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    with open(ATTENDANCE_FILE, "r") as f:
        try:
            return json.load(f)
        except:
            return []

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
        
        if known_face_encodings:
            # tolerance=0.5 makes recognition stricter/more accurate
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = 1.0 - face_distances[best_match_index]
        
        if name != "Unknown":
            log_attendance(name)
            
        detections.append({
            "name": name,
            "box": [int(top), int(right), int(bottom), int(left)],
            "confidence": float(confidence)
        })
        
    return {"detections": detections}

if __name__ == "__main__":
    # Use 127.0.0.1 for more reliable local connections on Windows
    uvicorn.run(app, host="127.0.0.1", port=8000)