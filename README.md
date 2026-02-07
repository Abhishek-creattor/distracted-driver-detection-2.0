# Distracted Driver Detection System (Offline + Unlimited + MongoDB)

Production-style, fully offline driver monitoring pipeline built with Python, OpenCV, MediaPipe, face-recognition (dlib), YOLO phone detection, and MongoDB.

## Why this stack
- **OpenCV**: camera ingestion + real-time visualization.
- **face_recognition (dlib)**: local face embeddings (128-D) for identity matching.
- **MediaPipe Face Mesh**: robust landmarks for EAR (drowsiness), MAR (yawning), and head-pose yaw.
- **YOLO (Ultralytics)**: local phone detection (`cell phone` class id 67).
- **MongoDB + pymongo**: unlimited local/self-hosted event and embedding storage.

No cloud API, no token limits, no rate limiting by external providers.

---

## 1) Text Architecture Diagram

```text
USB / IP Camera
    |
    v
OpenCV Frame Capture
    |
    +--> Face Embedding Extraction (face_recognition/dlib)
    |        |
    |        +--> MongoDB driver lookup (embedding distance match)
    |
    +--> MediaPipe Face Mesh
    |        |
    |        +--> EAR -> Drowsiness
    |        +--> MAR -> Yawning
    |        +--> Head Pose (yaw) -> Looking away
    |
    +--> YOLO phone detector
    |        |
    |        +--> Phone usage alert
    |
    v
Alert Fusion + Event Logging
    |
    v
MongoDB distraction_events collection
```

---

## 2) Project Structure

```text
.
├── config/
│   └── config.yaml
├── docs/
│   └── mongodb_schema_examples.json
├── models/
│   └── yolov8n.pt               # downloaded once (offline runtime after that)
├── scripts/
│   └── download_yolo_model.py
├── src/
│   └── ddd/
│       ├── __init__.py
│       ├── calibration.py       # per-driver EAR/MAR auto tuning
│       ├── cli.py               # CLI entrypoints
│       ├── config.py            # YAML config loader
│       ├── db.py                # MongoDB repo + indexes
│       ├── distraction.py       # EAR/MAR/head-pose
│       ├── face.py              # embedding extraction + matching
│       ├── monitor.py           # live monitoring + event logging
│       ├── phone_detection.py   # YOLO phone detection
│       └── register.py          # face registration
├── pyproject.toml
└── requirements.txt
```

---

## 3) Installation

### Prerequisites
- Python 3.9+
- MongoDB Community Server running locally or self-hosted
- Webcam

### Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Download YOLO model once
```bash
python scripts/download_yolo_model.py
```

---

## 4) Configuration

Edit `config/config.yaml`:

```yaml
mongo:
  uri: "mongodb://localhost:27017"
  db: "driver_monitoring"
camera:
  index: 0
face_recognition:
  distance_threshold: 0.50
alerts:
  min_eye_closed_frames: 20
  head_away_angle_deg: 25
phone_detection:
  confidence: 0.35
  yolov8_model_path: "models/yolov8n.pt"
calibration:
  frames: 120
```

---

## 5) CLI Usage

### Register driver
```bash
ddd-register --driver-id DRV001 --name "Amit Sharma" --config config/config.yaml
```
Press `c` to capture face.

### Calibrate driver thresholds (EAR/MAR)
```bash
ddd-calibrate --driver-id DRV001 --config config/config.yaml
```
The system automatically computes personalized thresholds from live baseline frames.

### Start monitoring
```bash
ddd-monitor --config config/config.yaml
```
Alerts are shown on-screen and logged to MongoDB.

---

## 6) MongoDB Schema (Examples)

See `docs/mongodb_schema_examples.json`.

Collections:
- `drivers`: identity, embedding vector, personalized thresholds.
- `distraction_events`: event stream (`drowsiness`, `head_away`, `yawning`, `phone_usage`) with UTC timestamps.

---

## 7) How “Unlimited Requests” is Achieved

- Entire inference path runs **locally** (dlib/MediaPipe/YOLO in-process).
- Identity matching is local vector-distance computation.
- MongoDB is local/self-hosted, horizontally scalable if needed.
- No SaaS API dependencies => no per-minute quotas, no paid request caps.

---

## 8) Real-Time Performance Tips

- Use lower camera resolution (`640x480`) for embedded devices.
- Run phone detector every N frames (e.g., every 3rd frame).
- Use GPU-enabled OpenCV/torch where available.
- Cache driver embeddings in memory and refresh periodically.
- Batch insert events or throttle duplicate alerts.

---

## 9) Optional Dashboard Integration

You can wrap MongoDB event streams with Flask/FastAPI endpoints and visualize:
- recent violations
- per-driver trend lines
- alert frequency heatmaps

---

## 10) Datasets for Evaluation

- StateFarm Distracted Driver Detection
- NTHU Driver Drowsiness Detection Dataset
- YawDD (Yawning Detection)
- Custom in-car captures for deployment calibration

