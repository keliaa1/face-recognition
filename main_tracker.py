# main_tracker.py
"""
Face Tracker with Servo Control
- Recognizes only the enrolled person (Kelia)
- Sends servo angle commands to ESP8266 over WiFi
"""

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import requests
import time
from pathlib import Path

# ============================================================
# CONFIGURATION - EDIT THESE
# ============================================================
ESP8266_IP = "10.12.74.37"       # <-- REPLACE with your ESP8266 IP
TARGET_NAME = "Kelia"              # <-- Your enrolled name
SIMILARITY_THRESHOLD = 0.55        # Adjust based on evaluate.py results
SMOOTHING_FACTOR = 0.15        # 0.0=no movement, 1.0=instant
FRAME_WIDTH = 640                  # Resize frame for speed
FRAME_HEIGHT = 480
# ============================================================

# ---- Load Face Database ----
DB_PATH = Path("data/db/face_db.npz")
if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found at {DB_PATH}. Run enrollment first.")

data = np.load(DB_PATH)
db_embeddings = {k: data[k].astype(np.float32) for k in data.files}

if TARGET_NAME not in db_embeddings:
    raise ValueError(f"'{TARGET_NAME}' not found in DB. Available: {list(db_embeddings.keys())}")

my_embedding = db_embeddings[TARGET_NAME]
print(f"Loaded embedding for '{TARGET_NAME}' (dim={my_embedding.size})")

# ---- Load ArcFace ONNX Model ----
ort_session = ort.InferenceSession(
    "models/embedder_arcface.onnx",
    providers=["CPUExecutionProvider"],
)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
print(f"ONNX model loaded. Input: {input_name}, Output: {output_name}")

# ---- MediaPipe FaceMesh ----
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ---- Haar Cascade ----
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade. Did you download the XML file?")

# ---- 5-point landmark indices ----
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291

# ---- Canonical 112x112 alignment targets ----
DST_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def get_5pt_landmarks(roi_bgr):
    """Extract 5 landmarks from a face ROI using MediaPipe."""
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    res = mp_face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    H, W = roi_bgr.shape[:2]
    idxs = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]
    pts = [[lm[i].x * W, lm[i].y * H] for i in idxs]
    return np.array(pts, dtype=np.float32)


def align_face(frame, kps):
    """Warp face to canonical 112x112 using 5 landmarks."""
    M, _ = cv2.estimateAffinePartial2D(kps, DST_POINTS, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)


def get_embedding(aligned_face):
    """Run ArcFace ONNX inference. Returns L2-normalized 512-D vector."""
    img = cv2.resize(aligned_face, (112, 112))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 128.0
    x = rgb[None, ...]  # NHWC (1, 112, 112, 3) -- matches our model
    y = ort_session.run([output_name], {input_name: x})[0]
    emb = y.reshape(-1).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


def send_servo_command(angle):
    """Send HTTP request to ESP8266 to move servo."""
    try:
        url = f"http://{ESP8266_IP}/MOVE?angle={angle}"
        requests.get(url, timeout=0.15)
    except Exception:
        pass  # Ignore network errors to keep video smooth


# ---- Main Loop ----
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

print(f"\nTracking '{TARGET_NAME}'. Press 'q' to quit.\n")

last_angle = 90
prev_time = time.time()
fps = 0.0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    vis = frame.copy()

    # Detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(70, 70))

    status_text = "Searching..."
    status_color = (128, 128, 128)

    if len(faces) > 0:
        # Pick largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]

        # Get landmarks from ROI
        roi = frame[y:y + h, x:x + w]
        kps = get_5pt_landmarks(roi)

        if kps is not None:
            # Convert landmarks to full-frame coordinates
            kps[:, 0] += x
            kps[:, 1] += y

            # Align & embed
            aligned = align_face(frame, kps)

            if aligned is not None:
                query_emb = get_embedding(aligned)

                # Cosine similarity (both vectors are L2-normalized)
                sim = float(np.dot(query_emb, my_embedding))
                is_me = sim >= SIMILARITY_THRESHOLD

                if is_me:
                    status_text = f"{TARGET_NAME} ({sim:.2f})"
                    status_color = (0, 255, 0)

                    # ---- SERVO CONTROL ----
                    face_center_x = x + w // 2
                    target_angle = int(np.interp(face_center_x, [0, FRAME_WIDTH], [180, 0]))

                    # Smooth movement
                    smoothed_angle = int(last_angle + (target_angle - last_angle) * SMOOTHING_FACTOR)
                    send_servo_command(smoothed_angle)
                    last_angle = smoothed_angle

                    # Draw servo angle
                    cv2.putText(vis, f"Servo: {smoothed_angle} deg", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    status_text = f"Unknown ({sim:.2f})"
                    status_color = (0, 0, 255)
                    # NOT ME -> do not move the servo

                # Draw bounding box and landmarks
                cv2.rectangle(vis, (x, y), (x + w, y + h), status_color, 2)
                for (px, py) in kps.astype(int):
                    cv2.circle(vis, (int(px), int(py)), 3, (0, 255, 255), -1)

                # Draw center crosshair
                cx = x + w // 2
                cv2.line(vis, (cx, y), (cx, y + h), (255, 0, 255), 1)

    # FPS
    frame_count += 1
    now = time.time()
    if now - prev_time >= 1.0:
        fps = frame_count / (now - prev_time)
        frame_count = 0
        prev_time = now

    # Status text
    cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, "q=quit", (10, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Face Tracker", vis)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()