from __future__ import annotations

import datetime
import os
import hmac
import base64
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
EVIDENCE_DIR = BASE_DIR / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

YOLO_PATH = MODEL_DIR / "yolov8n.pt"
RESNET_PATH = MODEL_DIR / "violence_resnet18.pth"

FLOW_THRESHOLD = 0.45
MOVE_THRESHOLD = 35
CONFIRM_SECONDS = 3
BUFFER_SIZE = 15
STAGE2_THRESHOLD = 0.85
DETECTION_EVERY_N_FRAMES = 3

app = Flask(__name__)

# Protect the public website with HTTP Basic Auth when deployed.
# Set TG_USERNAME and TG_PASSWORD as Render environment variables.
@app.before_request
def require_deployment_login():
    if os.getenv("RENDER"):
        expected_user = os.getenv("TG_USERNAME")
        expected_password = os.getenv("TG_PASSWORD")
        if not expected_user or not expected_password:
            return ("Server access is not configured. Set TG_USERNAME and TG_PASSWORD.", 503)
        header = request.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="TechGuard"'})
        try:
            decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
            user, password = decoded.split(":", 1)
        except Exception:
            return Response("Invalid credentials", 401, {"WWW-Authenticate": 'Basic realm="TechGuard"'})
        if not (hmac.compare_digest(user, expected_user) and hmac.compare_digest(password, expected_password)):
            return Response("Invalid credentials", 401, {"WWW-Authenticate": 'Basic realm="TechGuard"'})
    return None

state_lock = threading.Lock()
state = {
    "running": False,
    "source": "none",
    "status": "READY",
    "confidence": 0.0,
    "flow": 0.0,
    "fight_score": 0,
    "people": 0,
    "events": [],
    "error": None,
}
cap = None
worker = None
latest_jpeg = None
stop_event = threading.Event()

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = None
stage2 = None
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def log_event(message, level="info"):
    item = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "message": message,
        "level": level,
    }
    with state_lock:
        state["events"].insert(0, item)
        del state["events"][100:]

def load_models():
    global yolo, stage2
    if not YOLO_PATH.is_file():
        raise FileNotFoundError(f"Missing {YOLO_PATH}. Copy yolov8n.pt into the models folder.")
    if not RESNET_PATH.is_file():
        raise FileNotFoundError(f"Missing {RESNET_PATH}. Copy violence_resnet18.pth into the models folder.")
    log_event("Loading YOLOv8 model…")
    yolo = YOLO(str(YOLO_PATH))
    try:
        yolo.fuse()
    except Exception:
        pass
    log_event("YOLOv8 loaded.")
    log_event("Loading ResNet18 classifier…")
    stage2 = models.resnet18(weights=None)
    stage2.fc = nn.Linear(stage2.fc.in_features, 2)
    weights = torch.load(str(RESNET_PATH), map_location=device, weights_only=True)
    stage2.load_state_dict(weights)
    stage2.to(device)
    stage2.eval()
    log_event(f"ResNet18 loaded on {device}.")
    with state_lock:
        state["status"] = "READY"

def stage2_predict(frames):
    if not frames:
        return 0.0
    probs = []
    with torch.inference_mode():
        for frame in frames:
            # OpenCV frames are BGR; convert to RGB for the classifier.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).to(device)
            output = stage2(tensor)
            probability = torch.softmax(output, dim=1)
            probs.append(float(probability[0][1].item()))
    return float(np.mean(probs))

def camera_worker(source, is_file):
    global cap, latest_jpeg
    prev_gray = None
    prev_centers = []
    buffer = deque(maxlen=BUFFER_SIZE)
    suspicious_since = None
    last_alert = 0
    frame_count = 0
    cached_boxes = []
    cached_flow = 0.0
    cached_score = 0
    cached_stage1 = False
    cached_confidence = 0.0
    cached_violent = False

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    log_event("Test video finished.")
                else:
                    log_event("Camera frame could not be read.", "warning")
                break

            if frame.shape[1] > 960:
                scale = 960 / frame.shape[1]
                frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

            buffer.append(frame.copy())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

            if frame_count % DETECTION_EVERY_N_FRAMES == 0 or frame_count == 1:
                results = yolo.predict(source=frame, conf=0.4, classes=[0], verbose=False, imgsz=640)
                centers, boxes = [], []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
                        boxes.append((x1, y1, x2, y2))

                score = 0
                for i in range(min(len(centers), len(prev_centers))):
                    dx = centers[i][0] - prev_centers[i][0]
                    dy = centers[i][1] - prev_centers[i][1]
                    if (dx * dx + dy * dy) ** 0.5 > MOVE_THRESHOLD:
                        score += 1

                flow_energy = 0.0
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flow_energy = float(np.mean(magnitude))

                stage1 = flow_energy > FLOW_THRESHOLD or score >= 2
                confidence = 0.0
                violent = False
                now = time.time()
                if stage1:
                    if suspicious_since is None:
                        suspicious_since = now
                        log_event("Suspicious movement; running confirmation stage.")
                    elif now - suspicious_since >= CONFIRM_SECONDS and len(buffer) == BUFFER_SIZE:
                        confidence = stage2_predict(list(buffer))
                        if confidence > STAGE2_THRESHOLD:
                            violent = True
                else:
                    suspicious_since = None

                cached_boxes = boxes
                cached_flow = flow_energy
                cached_score = score
                cached_stage1 = stage1
                cached_confidence = confidence
                cached_violent = violent
                prev_gray = gray.copy()
                prev_centers = centers.copy()

            people = len(cached_boxes)
            if cached_violent:
                status_text = "POSSIBLE VIOLENCE - REVIEW"
                color = (0, 0, 255)
                if time.time() - last_alert > 2:
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    evidence_path = EVIDENCE_DIR / f"alert_{stamp}.jpg"
                    cv2.imwrite(str(evidence_path), frame)
                    log_event(f"Possible violence flagged; evidence saved: {evidence_path.name}", "alert")
                    last_alert = time.time()
            elif cached_stage1:
                status_text = "CHECKING - REVIEW"
                color = (0, 165, 255)
            else:
                status_text = "MONITORING"
                color = (0, 200, 0)

            for x1, y1, x2, y2 in cached_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status_text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"People: {people} | Violence score: {cached_confidence:.2f}", (18, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with state_lock:
                    latest_jpeg = encoded.tobytes()
                    state["people"] = people
                    state["confidence"] = round(cached_confidence, 3)
                    state["flow"] = round(cached_flow, 3)
                    state["fight_score"] = cached_score
                    state["status"] = status_text

            # Limit file playback from overwhelming the browser; camera feed remains responsive.
            time.sleep(0.01)
    except Exception as exc:
        log_event(f"Processing error: {exc}", "error")
        with state_lock:
            state["error"] = str(exc)
    finally:
        if cap is not None:
            cap.release()
        with state_lock:
            cap = None
            state["running"] = False
            if state["status"] not in ("POSSIBLE VIOLENCE - REVIEW", "CHECKING - REVIEW"):
                state["status"] = "STOPPED"
        log_event("Monitoring stopped.")

def start_monitoring(source, is_file=False):
    global cap, worker
    if yolo is None or stage2 is None:
        raise RuntimeError("AI models are not loaded. Check the startup log.")
    with state_lock:
        if state["running"]:
            raise RuntimeError("Monitoring is already running.")
    stop_event.clear()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        cap = None
        raise RuntimeError("Could not open the selected camera or video.")
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    with state_lock:
        state["running"] = True
        state["source"] = "video" if is_file else f"camera {source}"
        state["error"] = None
    log_event(f"Monitoring started: {state['source']}.")
    worker = threading.Thread(target=camera_worker, args=(source, is_file), daemon=True)
    worker.start()

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/status")
def api_status():
    with state_lock:
        return jsonify({**state, "events": list(state["events"])})

@app.post("/api/start-camera")
def api_start_camera():
    data = request.get_json(silent=True) or {}
    camera_id = int(data.get("camera_id", 0))
    try:
        start_monitoring(camera_id, is_file=False)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.post("/api/start-video")
def api_start_video():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Choose a video file first."}), 400
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        return jsonify({"ok": False, "error": "Supported formats: MP4, AVI, MOV, MKV."}), 400
    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    saved = upload_dir / f"test_{datetime.datetime.now():%Y%m%d_%H%M%S}{suffix}"
    file.save(saved)
    try:
        start_monitoring(str(saved), is_file=True)
        return jsonify({"ok": True, "filename": saved.name})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.post("/api/stop")
def api_stop():
    stop_event.set()
    with state_lock:
        running = state["running"]
    if not running:
        return jsonify({"ok": True, "message": "Already stopped."})
    return jsonify({"ok": True, "message": "Stop requested."})

@app.get("/video-feed")
def video_feed():
    def generate():
        while True:
            with state_lock:
                frame = latest_jpeg
            if frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.04)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Load models at import time so production WSGI servers (Gunicorn) load them too.
try:
    load_models()
except Exception as exc:
    log_event(f"Model loading failed: {exc}", "error")
    with state_lock:
        state["error"] = str(exc)
        state["status"] = "MODEL ERROR"
    print(f"MODEL ERROR: {exc}")
    print("Ensure both model files are available in the models folder.")

if __name__ == "__main__":
    # Local development only. Hosted deployment uses Gunicorn and the platform port.
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)
