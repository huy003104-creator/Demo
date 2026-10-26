TECHGUARD WEB + REAL AI BACKEND (LOCAL TEST VERSION)
====================================================

WHAT THIS DOES
- Web dashboard served by Flask.
- Uses your local YOLOv8n model for person detection.
- Uses your trained ResNet18 model for violence classification.
- Can process the laptop camera or an uploaded test video.
- Shows a processed MJPEG feed, detection metrics, and backend event log.
- Includes a basic visitor-entry table in the browser. This table is a prototype and is not saved to a secure database.

IMPORTANT LIMITS
- This does not identify a person's identity or automatically decide whether they may enter school.
- AI detections can be wrong. Have an authorised adult review any flagged footage.
- Use only camera feeds and videos you are authorised to process. Do not expose this development server to the public internet.
- The existing ResNet18 classifier's training labels and accuracy must be validated. A score is not proof of violence.
- This starter is intended for local development/testing, not live school deployment.

SETUP (Windows)
1. Extract this folder.
2. Inside TechGuardWeb_AI, create a folder named "models".
3. Copy your existing model files into models:
     models\yolov8n.pt
     models\violence_resnet18.pth
4. Open Terminal / PowerShell in the TechGuardWeb_AI folder.
5. Create a virtual environment (recommended):
     py -m venv .venv
     .venv\Scripts\activate
6. Install packages:
     python -m pip install --upgrade pip
     pip install -r requirements.txt
7. Start the website:
     python app.py
8. Open Chrome and go to:
     http://127.0.0.1:5000

TESTING
- Click "Start laptop camera" to test your webcam.
- Or choose an MP4/AVI/MOV/MKV file and click "Run AI on video".
- Click Stop to end processing.
- The first launch may take time while PyTorch and YOLO load.
- Evidence images are saved in the evidence folder when the current model pipeline flags possible violence.
- Uploaded videos are copied to uploads.

TROUBLESHOOTING
- Missing model: check the two filenames and the models folder.
- Import errors: ensure the virtual environment is active and requirements installed.
- Webcam won't open: close other apps using the camera. USB camera support can require changing camera ID in app.py.
- Model errors: send the full terminal traceback. PyTorch/torchvision versions must be compatible.
- The AI logic here is adapted from your Tkinter code but is not a line-by-line replacement. Compare its results with your original program on labelled test clips before relying on it.

VISITOR ENTRY
The visitor form is only a browser-session demonstration and does not persist data after refresh. For real use, it needs authenticated staff access, a database, audit logs, data retention rules, and school approval.

ONLINE DEPLOYMENT NOTES
=======================
This package includes a Procfile and render.yaml for a Render web service.
It uses Gunicorn and reads the host port from $PORT.

IMPORTANT:
- Public hosted use is protected by HTTP Basic Auth when running on Render.
- Set TG_USERNAME and TG_PASSWORD in Render's Environment settings before using the service.
- Do not commit model weights or real surveillance footage to a public GitHub repository.
- The current starter writes uploaded videos/evidence to local folders. Hosted ephemeral storage may be erased at restarts/redeploys. Add private object storage or a database only if you have approval and a retention policy.
- Test clips only at first. This is not yet validated for school deployment, and alerts require human review.
- Hosted server cannot automatically access a camera physically connected to a visitor's laptop. Users must upload a video, or you must separately build a secure camera-to-server stream.
- Model weights are not bundled in this zip. Place them in models/ locally; for hosted deployment, use a private storage/download workflow or a private repo with appropriate size limits, and do not expose them publicly.
