------> VisionTrack AI: Smart Attendance System <---------

VisionTrack AI is a state-of-the-art biometric attendance solution. It leverages Deep Learning to detect and recognize faces in real-time, automating the check-in process for modern workplaces and educational institutions.

✨ Key Features

🚀 Real-Time Face Recognition: Low-latency detection using Dlib's ResNet-128 model.

👤 Instant Enrollment: Register new users directly through the web dashboard with a single snapshot.

📊 Live Analytics Dashboard: Interactive UI showing real-time detection logs and daily attendance metrics.

🛠️ DevOps Ready: Fully containerized with Docker, featuring integrated monitoring via Prometheus.

📈 High Accuracy: Robust performance across varying lighting conditions using 128D facial embeddings.

🛠️ Tech Stack

Backend: Python, FastAPI, Uvicorn

Computer Vision: OpenCV, face_recognition (Dlib)

Frontend: Tailwind CSS, JavaScript (ES6)

Infrastructure: Docker, Docker Compose, Prometheus, Grafana

📂 Project Hierarchy

smart-attendance-app/
├── app.py                 # Backend API & AI processing logic
├── index.html             # Real-time monitoring dashboard
├── requirements.txt       # Python library dependencies
├── Dockerfile             # Backend container configuration
├── docker-compose.yml     # Multi-service orchestration
├── prometheus.yml         # Monitoring configuration
├── attendance.json        # Data storage (auto-generated)
└── known_faces/           # Database of registered facial images


🚀 Getting Started

1. Local Setup

Install Dependencies:

pip install -r requirements.txt


Run the Server:

python app.py


Open the Dashboard:
Simply open index.html in your web browser.

2. Docker Setup

Run the entire stack (API + Database + Monitoring) with:

docker-compose up --build


📖 Usage Guide

Enrollment: Enter a name in the "New Enrollment" panel and click Capture & Register.

Recognition: The system automatically scans the live feed. Recognized users are highlighted with a blue box.

Logging: Attendance is logged instantly to attendance.json and reflected on the sidebar history.

🗺️ Roadmap

[ ] Implement JWT-based authentication for the dashboard.

[ ] Integrate FAISS for large-scale facial search optimization.

[ ] Add Email/Slack notifications for unrecognized entries.

[ ] Develop an export feature for Excel/CSV reports.

📄 License

This project is licensed under the MIT License.

VisionTrack AI — The future of secure, automated attendance.

# Sonal Kumar