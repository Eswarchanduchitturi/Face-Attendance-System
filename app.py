from flask import Flask, render_template, Response, request, redirect, url_for, send_file, jsonify, session

import cv2
import sqlite3
import datetime
import os
import io
import math
import threading
import uuid
from collections import deque

import numpy as np
from PIL import Image
import time
from routes.training import create_training_blueprint



# System metrics
camera_status = "OFF"
fps_value = 0
frame_count = 0
start_time = time.time()

# Model performance metrics
success_recognition = 0
failed_recognition = 0
confidence_scores = []




db = sqlite3.connect("database/attendance.db")

db.execute("INSERT OR IGNORE INTO users VALUES (1,'Eswar','admin')")
db.execute("INSERT OR IGNORE INTO users VALUES (2,'Ravi','user')")

db.commit()
db.close()

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


blink_detected = False
head_moved = False
prev_face_x = None
camera = None
camera_enabled = False
attendance_status = ""
attendance_mode = "checkin"   # default mode
last_attendance_time = 0
ATTENDANCE_COOLDOWN = 3  # seconds






from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
from flask import session



# ------------------ Flask App ------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
# ------------------ Config ------------------
CAMERA_INDEX = 0
THRESHOLD = 58          # Balanced threshold for fewer false negatives in live recognition.
TOTAL_EMPLOYEES = 10    # change later (will auto-calc with users table)
RECOGNITION_WINDOW = 5
REQUIRED_STABLE_MATCHES = 3
VERIFY_MIN_CORR = 0.58
VERIFY_MIN_GAP = 0.02
MAX_VERIFY_SAMPLES_PER_USER = 8
VERIFY_MAX_LBP_DIST = 0.95
VERIFY_LBP_MARGIN = 0.01
VERIFY_MIN_VOTES = 1
FALLBACK_CONF_ACCEPT = 42.0
AUTO_THRESHOLD_ENABLED = True
AUTO_THRESHOLD_MIN = 45
AUTO_THRESHOLD_MAX = 70
AUTO_THRESHOLD_STEP_UP = 2
AUTO_THRESHOLD_STEP_DOWN = 1
AUTO_THRESHOLD_FAIL_TRIGGER = 10
AUTO_THRESHOLD_SUCCESS_TRIGGER = 6
RELAXED_MODE_FAIL_TRIGGER = 6
RELAXED_CONF_MARGIN = 4
RECOGNITION_HOLD_SECONDS = 1.5
LIVE_SECONDARY_VERIFY = False
MODEL_RELOAD_INTERVAL_SEC = 2.0
FACE_DETECT_SCALE = 0.6
RECOGNITION_DEBUG = False
MASK_AWARE_ENABLED = True
MASK_THRESHOLD_BOOST = 8
MASK_REQUIRED_STABLE_MATCHES = 4
MASK_MIN_UPPER_CORR = 0.50
MASK_MIN_UPPER_GAP = 0.015
MASK_MAX_UPPER_LBP_DIST = 1.20
MASK_MIN_UPPER_LBP_MARGIN = 0.005
IRIS_VERIFY_ENABLED = True
IRIS_MIN_SCORE = 0.26
IRIS_MIN_GAP = 0.015
IRIS_MIN_VOTES = 1

face_samples_cache = {}
face_samples_cache_model_mtime = None
recognizer_model_mtime = None

# ------------------ OpenCV Setup ------------------

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Only load model if it exists
if os.path.exists("TrainingImageLabel/Trainner.yml"):
    try:
        recognizer.read("TrainingImageLabel/Trainner.yml")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
else:
    print("⚠️  No trained model found. Enroll users to train the model.")

# ------------------ Database ------------------
def get_db():
    return sqlite3.connect("database/attendance.db")


def refresh_face_samples_cache():
    """Load a small set of enrolled face images per user for second-stage verification."""
    global face_samples_cache
    global face_samples_cache_model_mtime

    model_path = "TrainingImageLabel/Trainner.yml"
    model_mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else None

    if face_samples_cache and face_samples_cache_model_mtime == model_mtime:
        return

    new_cache = {}
    if os.path.exists("TrainingImage"):
        for filename in sorted(os.listdir("TrainingImage")):
            if not filename.endswith(".jpg") or not filename.startswith("User."):
                continue

            parts = filename.split(".")
            if len(parts) < 4:
                continue

            try:
                uid = int(parts[1])
            except ValueError:
                continue

            user_samples = new_cache.setdefault(uid, [])
            if len(user_samples) >= MAX_VERIFY_SAMPLES_PER_USER:
                continue

            img_path = os.path.join("TrainingImage", filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            upper = _extract_upper_face(img)
            iris_sig = _extract_iris_signature(img)
            user_samples.append({
                "img": img,
                "hist": _gray_hist(img),
                "lbp": _lbp_hist(img),
                "upper_hist": _gray_hist(upper),
                "upper_lbp": _lbp_hist(upper),
                "iris_sig": iris_sig
            })

    face_samples_cache = new_cache
    face_samples_cache_model_mtime = model_mtime


def verify_prediction_with_samples(face_roi_resized, predicted_user_id):
    """
    Verify predicted identity using histogram correlation against enrolled samples.
    Returns (is_verified, same_user_best_corr, other_user_best_corr).
    """
    refresh_face_samples_cache()

    predicted_samples = face_samples_cache.get(predicted_user_id, [])
    if not predicted_samples:
        return False, 0.0, 0.0, 999.0, 999.0, 999.0, 999.0, 0, None

    probe_hist = _gray_hist(face_roi_resized)
    probe_lbp = _lbp_hist(face_roi_resized)

    user_stats = {}
    for uid, samples in face_samples_cache.items():
        best_corr = -1.0
        best_lbp = 999.0
        for sample in samples:
            corr = cv2.compareHist(probe_hist, sample["hist"], cv2.HISTCMP_CORREL)
            if corr > best_corr:
                best_corr = corr

            lbp_dist = cv2.compareHist(probe_lbp, sample["lbp"], cv2.HISTCMP_CHISQR)
            if lbp_dist < best_lbp:
                best_lbp = lbp_dist
        user_stats[uid] = {"corr": best_corr, "lbp": best_lbp}

    best_corr_uid = max(user_stats.items(), key=lambda kv: kv[1]["corr"])[0]
    best_lbp_uid = min(user_stats.items(), key=lambda kv: kv[1]["lbp"])[0]
    pred_corr = user_stats[predicted_user_id]["corr"]
    pred_lbp = user_stats[predicted_user_id]["lbp"]

    other_corr = max(v["corr"] for uid, v in user_stats.items() if uid != predicted_user_id) if len(user_stats) > 1 else -1.0
    other_lbp = min(v["lbp"] for uid, v in user_stats.items() if uid != predicted_user_id) if len(user_stats) > 1 else 999.0

    votes = 0
    if best_corr_uid == predicted_user_id:
        votes += 1
    if best_lbp_uid == predicted_user_id:
        votes += 1

    is_verified = (
        pred_corr >= VERIFY_MIN_CORR
        and (pred_corr - other_corr) >= VERIFY_MIN_GAP
        and pred_lbp <= VERIFY_MAX_LBP_DIST
        and (other_lbp - pred_lbp) >= VERIFY_LBP_MARGIN
        and votes >= VERIFY_MIN_VOTES
    )

    return (
        is_verified,
        pred_corr,
        other_corr,
        pred_lbp,
        other_lbp,
        999.0,
        999.0,
        votes,
        {"corr_uid": best_corr_uid, "lbp_uid": best_lbp_uid}
    )


def _gray_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def _lbp_hist(img):
    # Basic 8-neighbor LBP encoding and normalized 256-bin histogram.
    center = img[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    lbp |= (img[:-2, :-2] >= center).astype(np.uint8) << 7
    lbp |= (img[:-2, 1:-1] >= center).astype(np.uint8) << 6
    lbp |= (img[:-2, 2:] >= center).astype(np.uint8) << 5
    lbp |= (img[1:-1, 2:] >= center).astype(np.uint8) << 4
    lbp |= (img[2:, 2:] >= center).astype(np.uint8) << 3
    lbp |= (img[2:, 1:-1] >= center).astype(np.uint8) << 2
    lbp |= (img[2:, :-2] >= center).astype(np.uint8) << 1
    lbp |= (img[1:-1, :-2] >= center).astype(np.uint8)
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def _extract_upper_face(img):
    """Use upper-face band (eyes/forehead) for mask-aware verification."""
    h = img.shape[0]
    upper_end = max(90, int(h * 0.62))
    upper = img[:upper_end, :]
    return cv2.resize(upper, (200, 120))


def _edge_density(img):
    edges = cv2.Canny(img, 70, 140)
    return float(np.count_nonzero(edges)) / float(edges.size)


def detect_probable_mask(face_roi_resized):
    """
    Heuristic mask detector:
    lower-half cloth masks are usually smoother and less textured than upper face.
    """
    h = face_roi_resized.shape[0]
    split = int(h * 0.58)
    upper = face_roi_resized[:split, :]
    lower = face_roi_resized[split:, :]

    upper_var = float(np.var(upper))
    lower_var = float(np.var(lower))
    upper_edges = _edge_density(upper)
    lower_edges = _edge_density(lower)

    texture_ratio = lower_var / (upper_var + 1e-6)
    edge_ratio = lower_edges / (upper_edges + 1e-6)

    return (
        texture_ratio < 0.72
        and edge_ratio < 0.72
        and lower_var < 1050
    )


def _extract_iris_signature(face_roi_resized):
    """
    Build a lightweight periocular signature from eye regions.
    Returns None if reliable eyes are not found.
    """
    upper = _extract_upper_face(face_roi_resized)
    eyes = eye_cascade.detectMultiScale(
        upper,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 12)
    )
    if len(eyes) == 0:
        return None

    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eye_features = []
    for (ex, ey, ew, eh) in sorted(eyes, key=lambda e: e[0]):
        eye_patch = upper[ey:ey + eh, ex:ex + ew]
        if eye_patch.size == 0:
            continue
        eye_patch = cv2.equalizeHist(eye_patch)
        eye_patch = cv2.resize(eye_patch, (64, 32))
        eye_features.append({
            "gray": _gray_hist(eye_patch),
            "lbp": _lbp_hist(eye_patch)
        })

    if len(eye_features) == 0:
        return None

    if len(eye_features) == 1:
        avg_gray = eye_features[0]["gray"]
        avg_lbp = eye_features[0]["lbp"]
    else:
        avg_gray = cv2.addWeighted(eye_features[0]["gray"], 0.5, eye_features[1]["gray"], 0.5, 0)
        avg_lbp = cv2.addWeighted(eye_features[0]["lbp"], 0.5, eye_features[1]["lbp"], 0.5, 0)

    return {"gray": avg_gray, "lbp": avg_lbp}


def verify_prediction_with_iris(face_roi_resized, predicted_user_id):
    """
    Verify identity with periocular/iris-like texture signatures.
    Returns (is_verified, pred_score, other_score, pred_lbp, other_lbp, votes, winners).
    """
    refresh_face_samples_cache()

    probe_sig = _extract_iris_signature(face_roi_resized)
    if probe_sig is None:
        return False, -1.0, -1.0, 999.0, 999.0, 0, {"mode": "iris_no_probe"}

    predicted_samples = face_samples_cache.get(predicted_user_id, [])
    if not predicted_samples:
        return False, -1.0, -1.0, 999.0, 999.0, 0, {"mode": "iris_no_samples"}

    user_stats = {}
    for uid, samples in face_samples_cache.items():
        best_score = -1.0
        best_corr = -1.0
        best_lbp = 999.0

        for sample in samples:
            iris_sig = sample.get("iris_sig")
            if iris_sig is None:
                continue

            corr = cv2.compareHist(probe_sig["gray"], iris_sig["gray"], cv2.HISTCMP_CORREL)
            lbp_dist = cv2.compareHist(probe_sig["lbp"], iris_sig["lbp"], cv2.HISTCMP_CHISQR)
            score = corr - (0.24 * lbp_dist)

            if score > best_score:
                best_score = score
            if corr > best_corr:
                best_corr = corr
            if lbp_dist < best_lbp:
                best_lbp = lbp_dist

        if best_score > -1.0:
            user_stats[uid] = {"score": best_score, "corr": best_corr, "lbp": best_lbp}

    if predicted_user_id not in user_stats:
        return False, -1.0, -1.0, 999.0, 999.0, 0, {"mode": "iris_pred_missing"}

    best_score_uid = max(user_stats.items(), key=lambda kv: kv[1]["score"])[0]
    best_lbp_uid = min(user_stats.items(), key=lambda kv: kv[1]["lbp"])[0]

    pred_score = user_stats[predicted_user_id]["score"]
    pred_lbp = user_stats[predicted_user_id]["lbp"]
    other_score = max(v["score"] for uid, v in user_stats.items() if uid != predicted_user_id) if len(user_stats) > 1 else -1.0
    other_lbp = min(v["lbp"] for uid, v in user_stats.items() if uid != predicted_user_id) if len(user_stats) > 1 else 999.0

    votes = 0
    if best_score_uid == predicted_user_id:
        votes += 1
    if best_lbp_uid == predicted_user_id:
        votes += 1

    is_verified = (
        pred_score >= IRIS_MIN_SCORE
        and (pred_score - other_score) >= IRIS_MIN_GAP
        and votes >= IRIS_MIN_VOTES
    )

    return (
        is_verified,
        pred_score,
        other_score,
        pred_lbp,
        other_lbp,
        votes,
        {"mode": "iris_verify", "score_uid": best_score_uid, "lbp_uid": best_lbp_uid}
    )


def verify_prediction_with_upper_samples(face_roi_resized, predicted_user_id):
    """
    Verify identity using only upper-face features for masked-face fallback.
    Returns (is_verified, pred_corr, other_corr, pred_lbp, other_lbp, votes, winners).
    """
    refresh_face_samples_cache()

    predicted_samples = face_samples_cache.get(predicted_user_id, [])
    if not predicted_samples:
        return False, 0.0, 0.0, 999.0, 999.0, 0, None

    upper_probe = _extract_upper_face(face_roi_resized)
    probe_hist = _gray_hist(upper_probe)
    probe_lbp = _lbp_hist(upper_probe)

    user_stats = {}
    for uid, samples in face_samples_cache.items():
        best_corr = -1.0
        best_lbp = 999.0
        for sample in samples:
            corr = cv2.compareHist(probe_hist, sample["upper_hist"], cv2.HISTCMP_CORREL)
            if corr > best_corr:
                best_corr = corr

            lbp_dist = cv2.compareHist(probe_lbp, sample["upper_lbp"], cv2.HISTCMP_CHISQR)
            if lbp_dist < best_lbp:
                best_lbp = lbp_dist

        user_stats[uid] = {"corr": best_corr, "lbp": best_lbp}

    best_corr_uid = max(user_stats.items(), key=lambda kv: kv[1]["corr"])[0]
    best_lbp_uid = min(user_stats.items(), key=lambda kv: kv[1]["lbp"])[0]
    pred_corr = user_stats[predicted_user_id]["corr"]
    pred_lbp = user_stats[predicted_user_id]["lbp"]
    other_corr = max(v["corr"] for uid, v in user_stats.items() if uid != predicted_user_id) if len(user_stats) > 1 else -1.0
    other_lbp = min(v["lbp"] for uid, v in user_stats.items() if uid != predicted_user_id) if len(user_stats) > 1 else 999.0

    votes = 0
    if best_corr_uid == predicted_user_id:
        votes += 1
    if best_lbp_uid == predicted_user_id:
        votes += 1

    is_verified = (
        pred_corr >= MASK_MIN_UPPER_CORR
        and (pred_corr - other_corr) >= MASK_MIN_UPPER_GAP
        and pred_lbp <= MASK_MAX_UPPER_LBP_DIST
        and (other_lbp - pred_lbp) >= MASK_MIN_UPPER_LBP_MARGIN
        and votes >= 1
    )

    return (
        is_verified,
        pred_corr,
        other_corr,
        pred_lbp,
        other_lbp,
        votes,
        {"corr_uid": best_corr_uid, "lbp_uid": best_lbp_uid}
    )


def _orb_desc(img):
    orb = cv2.ORB_create(300)
    _, des = orb.detectAndCompute(img, None)
    return des


def _orb_mean_distance(des1, des2):
    if des1 is None or des2 is None:
        return 999.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 999.0
    matches = sorted(matches, key=lambda m: m.distance)
    top = matches[:20]
    return float(sum(m.distance for m in top) / len(top))


def infer_user_id_from_face_image(image_rel_path):
    """
    Infer best user_id from saved check-in face image.
    Returns user_id or None when inference is ambiguous.
    """
    if not image_rel_path:
        return None

    full_path = os.path.join("static", image_rel_path)
    if not os.path.exists(full_path):
        return None

    refresh_face_samples_cache()
    if not face_samples_cache:
        return None

    gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    gray = cv2.resize(gray, (200, 200))

    probe_hist = _gray_hist(gray)
    probe_lbp = _lbp_hist(gray)

    user_scores = []
    for uid, samples in face_samples_cache.items():
        best_corr = -1.0
        best_lbp = 999.0
        for sample in samples:
            corr = cv2.compareHist(probe_hist, sample["hist"], cv2.HISTCMP_CORREL)
            lbp = cv2.compareHist(probe_lbp, sample["lbp"], cv2.HISTCMP_CHISQR)
            if corr > best_corr:
                best_corr = corr
            if lbp < best_lbp:
                best_lbp = lbp
        combined = best_corr - (0.30 * best_lbp)
        user_scores.append((uid, best_corr, best_lbp, combined))

    user_scores.sort(key=lambda x: x[3], reverse=True)
    if not user_scores:
        return None

    best_uid, best_corr, best_lbp, best_combined = user_scores[0]
    second_combined = user_scores[1][3] if len(user_scores) > 1 else -999.0

    if best_corr < 0.68:
        return None
    if (best_combined - second_combined) < 0.05:
        return None
    if best_lbp > 0.70:
        return None

    return best_uid

# ------------------ Attendance Logic ------------------
def mark_attendance(user_id, face_image):
    global attendance_mode

    print(f"\n{'='*60}")
    print(f"[MARK_ATTENDANCE] Function called with:")
    print(f"  user_id parameter: {user_id} (type: {type(user_id)})")
    print(f"  attendance_mode: {attendance_mode}")
    print(f"{'='*60}")

    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")
    day = now.strftime("%A")
    time_now_file = now.strftime("%H-%M-%S")
    time_now_db = now.strftime("%H:%M:%S")

    db = get_db()
    cur = db.cursor()

    # CRITICAL FIX: VALIDATE USER EXISTS
    user_check = cur.execute("""
        SELECT id, name FROM users WHERE id = ?
    """, (user_id,)).fetchone()
    
    if not user_check:
        db.close()
        print(f"❌ [ERROR] Recognized user_id {user_id} does NOT exist in database!")
        print(f"   Valid user IDs in DB:")
        db = get_db()
        valid_ids = db.execute("SELECT id, name FROM users").fetchall()
        for vid, vname in valid_ids:
            print(f"      - ID {vid}: {vname}")
        db.close()
        return f"invalid_user_{user_id}"
    
    user_name = user_check[1]
    print(f"✅ User validation successful: ID={user_id}, Name={user_name}")

    os.makedirs("static/attendance_images", exist_ok=True)

    # CHECK-IN
    if attendance_mode == "checkin":
        cur.execute("""
            SELECT id FROM attendance
            WHERE emp_id=? AND date=? AND checkout_time IS NULL
            ORDER BY id DESC LIMIT 1
        """, (user_id, today))

        open_session = cur.fetchone()

        if open_session:
            db.close()
            print(f"⚠️  User {user_id} already checked in today")
            return "already_checked_in"

        filename = f"checkin_{user_id}_{today}_{time_now_file}.jpg"
        full_path = os.path.join("static/attendance_images", filename)
        cv2.imwrite(full_path, face_image)
        db_path = f"attendance_images/{filename}"

        cur.execute("""
            INSERT INTO attendance 
            (emp_id, date, day, checkin_time, status, checkin_image)
            VALUES (?, ?, ?, ?, 'Present', ?)
        """, (user_id, today, day, time_now_db, db_path))

        db.commit()
        db.close()
        print(f"✅ Check-in recorded for User {user_id} ({user_name})")
        print(f"   Date: {today}, Time: {time_now_db}")
        return f"checkin_success_{user_name}"

    # CHECK-OUT
    elif attendance_mode == "checkout":
        cur.execute("""
            SELECT id, checkin_time
            FROM attendance
            WHERE emp_id=? AND date=? AND checkout_time IS NULL
            ORDER BY id DESC LIMIT 1
        """, (user_id, today))

        open_session = cur.fetchone()

        if not open_session:
            db.close()
            print(f"⚠️  No open session for User {user_id} today")
            return "already_checked_out"

        record_id, checkin_time = open_session
        login_time = datetime.datetime.strptime(checkin_time, "%H:%M:%S")
        logout_time = datetime.datetime.strptime(time_now_db, "%H:%M:%S")
        worked_hours = (logout_time - login_time).total_seconds() / 3600

        filename = f"checkout_{user_id}_{today}_{time_now_file}.jpg"
        full_path = os.path.join("static/attendance_images", filename)
        cv2.imwrite(full_path, face_image)
        db_path = f"attendance_images/{filename}"

        cur.execute("""
            UPDATE attendance
            SET checkout_time=?, worked_hours=?, checkout_image=?
            WHERE id=?
        """, (time_now_db, worked_hours, db_path, record_id))

        db.commit()
        db.close()
        print(f"✅ Check-out recorded for User {user_id} ({user_name})")
        print(f"   Date: {today}, Time: {time_now_db}, Worked: {worked_hours:.2f} hrs")
        return f"checkout_success_{user_name}"
    
    db.close()
    return "error_invalid_mode"


def get_attendance_stats():
    db = get_db()

    today_count = db.execute("""
        SELECT COUNT(DISTINCT user_id)
        FROM attendance
        WHERE date = date('now')
    """).fetchone()[0]

    total_count = db.execute("""
        SELECT COUNT(*)
        FROM attendance
    """).fetchone()[0]

    db.close()

    percentage = 0
    if TOTAL_EMPLOYEES > 0:
        percentage = round((today_count / TOTAL_EMPLOYEES) * 100, 2)

    return today_count, total_count, percentage

# ------------------ Video Streaming ------------------
def generate_frames():
    global attendance_status
    global success_recognition
    global failed_recognition
    global camera
    global last_attendance_time

    global recognizer_model_mtime

    # Keep a short rolling history so attendance is recorded only after stable identity.
    recognition_history = deque(maxlen=RECOGNITION_WINDOW)
    last_model_reload = 0.0
    dynamic_threshold = THRESHOLD
    no_match_streak = 0
    stable_success_streak = 0
    hold_until = 0.0
    hold_user_name = ""
    hold_status_text = ""
    hold_status_subtext = ""
    hold_status_color = (0, 255, 0)

    if camera is None:
        if os.name == "nt":
            camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        else:
            camera = cv2.VideoCapture(CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while camera_enabled:

        success, frame = camera.read()
        if not success:
            break

        # Keep live recognition orientation consistent with enrollment capture.
        # Mirroring here causes model mismatch if enrollment images are not mirrored.

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (0, 0), fx=FACE_DETECT_SCALE, fy=FACE_DETECT_SCALE)
        faces_small = face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(56, 56)
        )
        faces = []
        for (sx, sy, sw, sh) in faces_small:
            x = int(sx / FACE_DETECT_SCALE)
            y = int(sy / FACE_DETECT_SCALE)
            w = int(sw / FACE_DETECT_SCALE)
            h = int(sh / FACE_DETECT_SCALE)
            faces.append((x, y, w, h))
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        total_faces_detected = len(faces)
        if len(faces) > 1:
            faces = faces[:1]

        status_text = "Detecting Face..."
        status_subtext = "Keep one face centered for quick attendance."
        status_color = (0, 200, 255)
        user_name = ""
        active_threshold = dynamic_threshold if AUTO_THRESHOLD_ENABLED else THRESHOLD

        recognized_faces = []

        # Reload model only when needed to avoid frame lag.
        model_loaded = False
        try:
            model_path = "TrainingImageLabel/Trainner.yml"
            if os.path.exists(model_path):
                mtime = os.path.getmtime(model_path)
                now = time.time()
                if (
                    recognizer_model_mtime is None
                    or mtime != recognizer_model_mtime
                    or (now - last_model_reload) >= MODEL_RELOAD_INTERVAL_SEC
                ):
                    recognizer.read(model_path)
                    recognizer_model_mtime = mtime
                    last_model_reload = now
                model_loaded = True
        except Exception as e:
            print(f"Warning: Could not reload recognizer model: {e}")

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            # Normalize illumination to reduce false "not recognized" in uneven lighting.
            face_roi = cv2.equalizeHist(face_roi)

            # Skip prediction if model not available
            if not model_loaded:
                box_color = (0, 140, 255)
                label = "No Model - Enroll Users First"
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                continue
            
            # CRITICAL: Resize face to match training image dimensions (200x200)
            face_roi_resized = cv2.resize(face_roi, (200, 200))
            mask_suspected = MASK_AWARE_ENABLED and detect_probable_mask(face_roi_resized)
            
            user_id, conf = recognizer.predict(face_roi_resized)
            if RECOGNITION_DEBUG:
                print(f"[RECOGNITION] Predicted user_id: {user_id}, confidence: {conf:.2f}")

            box_color = (0, 0, 255)
            label = f"Unknown ({conf:.1f})"
            effective_threshold = active_threshold
            is_mask_mode = False

            conf_candidate = conf < active_threshold
            if (
                MASK_AWARE_ENABLED
                and mask_suspected
                and conf < (active_threshold + MASK_THRESHOLD_BOOST)
            ):
                conf_candidate = True

            if conf_candidate:
                db = get_db()
                user_check = db.execute("""
                    SELECT id, name FROM users WHERE id = ?
                """, (user_id,)).fetchone()
                db.close()

                if user_check:
                    detected_id = user_check[0]
                    detected_name = user_check[1]

                    # Default: in live mode prefer speed/reliability with LBPH + stable frame voting.
                    is_verified = True
                    votes = 1
                    winners = {"mode": "lbph_stable"}
                    same_corr = other_corr = same_lbp = other_lbp = same_orb = other_orb = 0.0

                    # Optional strict secondary verification.
                    if LIVE_SECONDARY_VERIFY:
                        (
                            is_verified,
                            same_corr,
                            other_corr,
                            same_lbp,
                            other_lbp,
                            same_orb,
                            other_orb,
                            votes,
                            winners
                        ) = verify_prediction_with_samples(face_roi_resized, detected_id)

                    fallback_ok = (conf <= FALLBACK_CONF_ACCEPT and votes >= 1)
                    relaxed_ok = (
                        total_faces_detected == 1
                        and no_match_streak >= RELAXED_MODE_FAIL_TRIGGER
                        and conf <= (active_threshold - RELAXED_CONF_MARGIN)
                        and votes >= 1
                    )

                    accepted = (is_verified or fallback_ok or relaxed_ok)
                    iris_verified = False

                    if (
                        not accepted
                        and MASK_AWARE_ENABLED
                        and mask_suspected
                        and conf < (active_threshold + MASK_THRESHOLD_BOOST)
                    ):
                        (
                            upper_verified,
                            upper_same_corr,
                            upper_other_corr,
                            upper_same_lbp,
                            upper_other_lbp,
                            upper_votes,
                            upper_winners
                        ) = verify_prediction_with_upper_samples(face_roi_resized, detected_id)

                        if upper_verified:
                            accepted = True
                            is_mask_mode = True
                            effective_threshold = active_threshold + MASK_THRESHOLD_BOOST
                            winners = {
                                "mode": "mask_upper_verify",
                                "upper_votes": upper_votes,
                                "upper_winners": upper_winners,
                                "upper_corr_same": round(upper_same_corr, 4),
                                "upper_corr_other": round(upper_other_corr, 4),
                                "upper_lbp_same": round(upper_same_lbp, 4),
                                "upper_lbp_other": round(upper_other_lbp, 4),
                            }

                    if not accepted and IRIS_VERIFY_ENABLED:
                        (
                            iris_verified,
                            iris_pred_score,
                            iris_other_score,
                            iris_pred_lbp,
                            iris_other_lbp,
                            iris_votes,
                            iris_winners
                        ) = verify_prediction_with_iris(face_roi_resized, detected_id)

                        if iris_verified:
                            accepted = True
                            is_mask_mode = bool(mask_suspected)
                            if mask_suspected:
                                effective_threshold = active_threshold + MASK_THRESHOLD_BOOST
                            winners = {
                                "mode": "iris_verify",
                                "iris_votes": iris_votes,
                                "iris_pred_score": round(iris_pred_score, 4),
                                "iris_other_score": round(iris_other_score, 4),
                                "iris_pred_lbp": round(iris_pred_lbp, 4),
                                "iris_other_lbp": round(iris_other_lbp, 4),
                                "iris_winners": iris_winners
                            }

                    if accepted:
                        box_color = (0, 255, 0)
                        if iris_verified and is_mask_mode:
                            label = f"ID: {detected_id} - {detected_name} [MASK+IRIS] ({conf:.1f})"
                        elif iris_verified:
                            label = f"ID: {detected_id} - {detected_name} [IRIS] ({conf:.1f})"
                        elif is_mask_mode:
                            label = f"ID: {detected_id} - {detected_name} [MASK] ({conf:.1f})"
                        else:
                            label = f"ID: {detected_id} - {detected_name} ({conf:.1f})"
                        if RECOGNITION_DEBUG:
                            print(
                                f"[MATCH] User {detected_id} ({detected_name}) conf={conf:.2f} "
                                f"corr_same={same_corr:.3f} corr_other={other_corr:.3f} "
                                f"lbp_same={same_lbp:.3f} lbp_other={other_lbp:.3f} "
                                f"orb_same={same_orb:.2f} orb_other={other_orb:.2f} "
                                f"votes={votes} winners={winners} fallback={fallback_ok} relaxed={relaxed_ok}"
                            )

                        face_image = frame[y:y+h, x:x+w].copy()
                        recognized_faces.append({
                            "user_id": user_id,
                            "user_name": detected_name,
                            "conf": conf,
                            "face_image": face_image,
                            "effective_threshold": effective_threshold,
                            "mask_mode": is_mask_mode
                        })
                    else:
                        box_color = (0, 165, 255)
                        label = f"Uncertain match ({conf:.1f})"
                        if RECOGNITION_DEBUG:
                            print(
                                f"[REJECTED] Pred={detected_id} conf={conf:.2f} "
                                f"corr_same={same_corr:.3f} corr_other={other_corr:.3f} "
                                f"lbp_same={same_lbp:.3f} lbp_other={other_lbp:.3f} "
                                f"orb_same={same_orb:.2f} orb_other={other_orb:.2f} "
                                f"votes={votes} winners={winners}"
                            )
                else:
                    label = f"Invalid ID {user_id} ({conf:.1f})"

            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

            # High-contrast label background for readability.
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            label_x1 = x
            label_y1 = max(8, y - th - 16)
            label_x2 = min(frame.shape[1] - 8, x + tw + 14)
            label_y2 = max(28, y - 4)
            cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (12, 20, 28), -1)
            cv2.putText(frame,
                        label,
                        (x + 6, label_y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        box_color,
                        2)

        current_time = time.time()

        if len(recognized_faces) == 1:
            best_match = recognized_faces[0]
            user_name = best_match["user_name"]
            required_stable_matches = (
                MASK_REQUIRED_STABLE_MATCHES
                if best_match.get("mask_mode")
                else REQUIRED_STABLE_MATCHES
            )
            recognition_history.append({
                "user_id": best_match["user_id"],
                "user_name": user_name,
                "conf": best_match["conf"],
                "face_image": best_match["face_image"],
                "effective_threshold": best_match.get("effective_threshold", active_threshold),
                "mask_mode": best_match.get("mask_mode", False)
            })

            stable_hits = [
                h for h in recognition_history
                if (
                    h["user_id"] == best_match["user_id"]
                    and h["conf"] < h.get("effective_threshold", active_threshold)
                    and h.get("mask_mode", False) == best_match.get("mask_mode", False)
                )
            ]

            if len(stable_hits) >= required_stable_matches:
                # Use the best-quality frame among stable hits.
                final_hit = min(stable_hits, key=lambda h: h["conf"])
                status_text = f"Verified: {final_hit['user_name']}"
                if final_hit.get("mask_mode"):
                    status_subtext = "Masked-face verified. Attendance confirmed."
                else:
                    status_subtext = "Attendance confirmed successfully."
                status_color = (0, 255, 0)

                print(f"\n[ATTENDANCE_RECORD] Stable match confirmed:")
                print(f"  User ID: {final_hit['user_id']}")
                print(f"  User Name: {final_hit['user_name']}")
                print(f"  Best Confidence: {final_hit['conf']:.2f}")
                print(f"  Stable Frames: {len(stable_hits)}/{required_stable_matches}")
                print(f"  THRESHOLD: {active_threshold}")

                if current_time - last_attendance_time > ATTENDANCE_COOLDOWN:
                    result = mark_attendance(final_hit["user_id"], final_hit["face_image"])
                    print(f"  Attendance recording result: {result}")
                    attendance_status = result
                    success_recognition += 1
                    last_attendance_time = current_time
                    recognition_history.clear()
                    no_match_streak = 0
                    stable_success_streak += 1
                    if (
                        AUTO_THRESHOLD_ENABLED
                        and stable_success_streak >= AUTO_THRESHOLD_SUCCESS_TRIGGER
                        and dynamic_threshold > AUTO_THRESHOLD_MIN
                    ):
                        dynamic_threshold = max(AUTO_THRESHOLD_MIN, dynamic_threshold - AUTO_THRESHOLD_STEP_DOWN)
                        stable_success_streak = 0
                hold_until = current_time + RECOGNITION_HOLD_SECONDS
                hold_user_name = final_hit["user_name"]
                hold_status_text = f"Verified: {final_hit['user_name']}"
                if final_hit.get("mask_mode"):
                    hold_status_subtext = "Masked-face verified. Attendance confirmed."
                else:
                    hold_status_subtext = "Attendance confirmed successfully."
                hold_status_color = (0, 255, 0)
            else:
                status_text = f"Verifying {user_name}... ({len(stable_hits)}/{required_stable_matches})"
                if best_match.get("mask_mode"):
                    status_subtext = "Mask detected: hold still slightly longer."
                else:
                    status_subtext = "Hold still for a second."
                status_color = (0, 200, 255)
                no_match_streak = 0
        elif total_faces_detected > 1:
            status_text = "Multiple recognized faces - show one face only"
            status_subtext = "Ask others to step out of frame."
            status_color = (0, 140, 255)
            attendance_status = "multiple_faces"
            recognition_history.clear()
            stable_success_streak = 0
        elif len(faces) > 0:
            status_text = "Face not recognized"
            status_subtext = "Try better light and look straight at camera."
            status_color = (0, 0, 255)
            attendance_status = "failed"
            failed_recognition += 1
            recognition_history.clear()
            stable_success_streak = 0
            if total_faces_detected == 1:
                no_match_streak += 1
                if (
                    AUTO_THRESHOLD_ENABLED
                    and no_match_streak >= AUTO_THRESHOLD_FAIL_TRIGGER
                    and dynamic_threshold < AUTO_THRESHOLD_MAX
                ):
                    dynamic_threshold = min(AUTO_THRESHOLD_MAX, dynamic_threshold + AUTO_THRESHOLD_STEP_UP)
                    no_match_streak = 0
        else:
            recognition_history.clear()
            stable_success_streak = 0

        # Smooth UI: keep verified identity visible for a short time to prevent flicker.
        if current_time < hold_until:
            status_text = hold_status_text or f"Verified: {hold_user_name}"
            status_subtext = hold_status_subtext or "Attendance confirmed successfully."
            status_color = hold_status_color

        # Centered, auto-sized status banner with stronger contrast.
        fh, fw = frame.shape[:2]
        title_font = cv2.FONT_HERSHEY_DUPLEX
        sub_font = cv2.FONT_HERSHEY_DUPLEX
        title_scale = 0.92
        sub_scale = 0.58

        (tw, th), _ = cv2.getTextSize(status_text, title_font, title_scale, 2)
        (sw, sh), _ = cv2.getTextSize(status_subtext, sub_font, sub_scale, 1)

        pad_x = 28
        panel_w = max(tw, sw) + (pad_x * 2)
        panel_h = th + sh + 40
        panel_w = min(panel_w, fw - 24)

        panel_x1 = max(12, (fw - panel_w) // 2)
        panel_y1 = 14
        panel_x2 = min(fw - 12, panel_x1 + panel_w)
        panel_y2 = min(fh - 12, panel_y1 + panel_h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (8, 16, 24), -1)
        frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (160, 186, 206), 1)

        title_x = panel_x1 + max(18, (panel_w - tw) // 2)
        title_y = panel_y1 + th + 12
        sub_x = panel_x1 + max(18, (panel_w - sw) // 2)
        sub_y = title_y + 28

        cv2.putText(frame, status_text, (title_x, title_y), title_font, title_scale, (0, 0, 0), 4)
        cv2.putText(frame, status_text, (title_x, title_y), title_font, title_scale, status_color, 2)
        cv2.putText(frame, status_subtext, (sub_x, sub_y), sub_font, sub_scale, (224, 235, 245), 1)
        cv2.putText(
            frame,
            f"Auto threshold: {active_threshold}",
            (panel_x1 + 16, panel_y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (168, 190, 210),
            1
        )

        _, buffer = cv2.imencode(".jpg", frame)

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    # Release camera when stopped
    if camera is not None:
        camera.release()
        camera = None


        
def train_model():
    """Train face recognition model - uses pre-detected face ROIs with consistent sizing"""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # CRITICAL: All images must be resized to same dimensions for LBPH training
    FACE_WIDTH = 200
    FACE_HEIGHT = 200

    faces = []
    ids = []
    MIN_IMAGES_PER_USER = 8

    if not os.path.exists("TrainingImage"):
        os.makedirs("TrainingImage")
        print("ERROR: TrainingImage folder was empty. Please enroll users first.")
        return False

    image_files = [f for f in os.listdir("TrainingImage") if f.endswith(".jpg")]
    if len(image_files) == 0:
        print("ERROR: No training images found. Please enroll users first.")
        return False

    print(f"[TRAINING] Processing {len(image_files)} images...")

    db = get_db()
    valid_user_ids = set(row[0] for row in db.execute("SELECT id FROM users").fetchall())
    db.close()

    image_paths = [os.path.join("TrainingImage", f) for f in image_files]
    user_face_count = {}

    for image_path in image_paths:
        try:
            # Load image as grayscale
            gray_img = Image.open(image_path).convert("L")
            img_np = np.array(gray_img, "uint8")

            # Verify minimal size
            if img_np.size < 100:
                continue

            filename = os.path.split(image_path)[-1]
            parts = filename.split(".")
            if len(parts) < 4 or parts[0] != "User":
                continue

            try:
                user_id = int(parts[1])
            except ValueError:
                continue

            if user_id not in valid_user_ids:
                continue

            # CRITICAL: Resize to standard dimensions
            resized_img = cv2.resize(img_np, (FACE_WIDTH, FACE_HEIGHT))
            
            faces.append(resized_img)
            ids.append(user_id)
            # Add a mirrored copy so prediction remains stable even if camera feed is flipped.
            faces.append(cv2.flip(resized_img, 1))
            ids.append(user_id)
            
            user_face_count[user_id] = user_face_count.get(user_id, 0) + 1
                
        except Exception:
            continue

    if len(faces) == 0:
        print(f"❌ ERROR: No valid faces found for training!")
        return False

    # Avoid training identities that have too few samples (high mismatch risk).
    eligible_user_ids = {uid for uid, count in user_face_count.items() if count >= MIN_IMAGES_PER_USER}
    if not eligible_user_ids:
        print(f"❌ ERROR: No users have enough samples. Need at least {MIN_IMAGES_PER_USER} images per user.")
        return False

    filtered_faces = []
    filtered_ids = []
    for face_img, uid in zip(faces, ids):
        if uid in eligible_user_ids:
            filtered_faces.append(face_img)
            filtered_ids.append(uid)

    skipped_users = {uid: cnt for uid, cnt in user_face_count.items() if uid not in eligible_user_ids}
    if skipped_users:
        print(f"⚠️ Skipping users with insufficient samples (<{MIN_IMAGES_PER_USER}): {skipped_users}")

    faces = filtered_faces
    ids = filtered_ids

    if len(faces) == 0:
        print("❌ ERROR: No eligible training samples after filtering.")
        return False

    print(f"[TRAINING] Training model with {len(faces)} samples from {len(eligible_user_ids)} users...")

    try:
        recognizer.train(faces, np.array(ids, dtype=np.int32))
        
        os.makedirs("TrainingImageLabel", exist_ok=True)
        recognizer.save("TrainingImageLabel/Trainner.yml")
        
        print(f"✅ Training complete!")
        return True
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        return False


def reload_recognizer_from_disk():
    """Reload latest trained model into memory."""
    global recognizer

    model_path = "TrainingImageLabel/Trainner.yml"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found on disk.")

    new_recognizer = cv2.face.LBPHFaceRecognizer_create()
    new_recognizer.read(model_path)
    recognizer = new_recognizer
    refresh_face_samples_cache()


class TrainingJobManager:
    """Simple in-process background training job manager."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}
        self._active_job_id = None

    def start_job(self, include_smoke_test=False):
        with self._lock:
            if self._active_job_id:
                active = self._jobs.get(self._active_job_id, {})
                if active.get("status") in ("queued", "running"):
                    return self._active_job_id

            job_id = str(uuid.uuid4())
            self._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "progress": 0,
                "stage": "queued",
                "message": "Training job queued",
                "include_smoke_test": include_smoke_test,
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "error": None,
            }
            self._active_job_id = job_id

        worker = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            daemon=True
        )
        worker.start()
        return job_id

    def get_job(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def get_active_job(self):
        with self._lock:
            if not self._active_job_id:
                return None
            active = self._jobs.get(self._active_job_id)
            return dict(active) if active else None

    def _update_job(self, job_id, **updates):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(updates)

    def _run_job(self, job_id):
        try:
            self._update_job(
                job_id,
                status="running",
                stage="precheck",
                progress=10,
                message="Validating training dataset",
                started_at=time.time(),
            )

            success = train_model()
            if not success:
                raise RuntimeError("Training failed. Check image quality and sample count per user.")

            self._update_job(
                job_id,
                stage="model_reload",
                progress=80,
                message="Reloading recognition model",
            )
            reload_recognizer_from_disk()

            job = self.get_job(job_id) or {}
            if job.get("include_smoke_test"):
                self._update_job(
                    job_id,
                    stage="smoke_test",
                    progress=92,
                    message="Running model smoke test",
                )
                if not os.path.exists("TrainingImageLabel/Trainner.yml"):
                    raise RuntimeError("Model smoke test failed: model file missing.")

            self._update_job(
                job_id,
                status="success",
                stage="completed",
                progress=100,
                message="Training completed successfully",
                finished_at=time.time(),
            )
        except Exception as e:
            self._update_job(
                job_id,
                status="fail",
                stage="failed",
                progress=100,
                message="Training failed",
                error=str(e),
                finished_at=time.time(),
            )
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None


training_jobs = TrainingJobManager()
app.register_blueprint(create_training_blueprint(training_jobs))

def eye_aspect_ratio(landmarks, left_eye, right_eye):
    def distance(p1, p2):
        return math.dist((p1.x, p1.y), (p2.x, p2.y))

    left_ear = (
        distance(landmarks[left_eye[1]], landmarks[left_eye[5]]) +
        distance(landmarks[left_eye[2]], landmarks[left_eye[4]])
    ) / (2.0 * distance(landmarks[left_eye[0]], landmarks[left_eye[3]]))

    right_ear = (
        distance(landmarks[right_eye[1]], landmarks[right_eye[5]]) +
        distance(landmarks[right_eye[2]], landmarks[right_eye[4]])
    ) / (2.0 * distance(landmarks[right_eye[0]], landmarks[right_eye[3]]))

    return (left_ear + right_ear) / 2.0

def admin_required():
    return session.get("admin", False)

def admin():
    if not admin_required():
        return redirect("/login")
    return render_template("admin.html")





# ------------------ Routes ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ---------------- COMBINED ATTENDANCE ----------------
@app.route("/attendance")
def attendance():

    db = get_db()

    # Fetch all attendance records with user role
    rows_raw = db.execute("""
        SELECT attendance.id,
               users.id,
               users.name,
               attendance.date,
               attendance.day,
               attendance.checkin_time,
               attendance.checkout_time,
               attendance.worked_hours,
               attendance.checkin_image,
               attendance.checkout_image,
               users.role
        FROM attendance
        INNER JOIN users 
            ON CAST(attendance.emp_id AS INTEGER) = users.id
        ORDER BY attendance.date DESC, attendance.checkin_time DESC
    """).fetchall()

    users_map = {
        r[0]: {"name": r[1], "role": r[2]}
        for r in db.execute("SELECT id, name, role FROM users").fetchall()
    }

    corrected_rows = []
    infer_cache = {}
    for row in rows_raw:
        row_list = list(row)
        checkin_image = row_list[8]
        if checkin_image:
            if checkin_image not in infer_cache:
                infer_cache[checkin_image] = infer_user_id_from_face_image(checkin_image)
            inferred_uid = infer_cache[checkin_image]
            if inferred_uid and inferred_uid in users_map and inferred_uid != row_list[1]:
                row_list[1] = inferred_uid
                row_list[2] = users_map[inferred_uid]["name"]
                row_list[10] = users_map[inferred_uid]["role"]
        corrected_rows.append(tuple(row_list))

    # Total users (all roles)
    total_users = db.execute("""
        SELECT COUNT(*) FROM users
    """).fetchone()[0]

    today = datetime.date.today().isoformat()

    # Today's attendance count (all roles)
    today_count = db.execute("""
        SELECT COUNT(DISTINCT attendance.emp_id)
        FROM attendance
        INNER JOIN users 
            ON CAST(attendance.emp_id AS INTEGER) = users.id
        WHERE attendance.date=?
    """, (today,)).fetchone()[0]

    total_count = len(corrected_rows)

    percentage = 0
    if total_users > 0:
        percentage = round((today_count / total_users) * 100, 2)

    db.close()

    return render_template(
        "attendance.html",
        rows=corrected_rows,
        total_users=total_users,
        today_count=today_count,
        total_count=total_count,
        percentage=percentage
    )

# ---------------- REDIRECTS FOR OLD ROUTES ----------------
@app.route("/attendance/students")
def student_attendance():
    return redirect("/attendance")

@app.route("/attendance/faculty")
def faculty_attendance():
    return redirect("/attendance")
@app.route("/attendance/analytics")
def attendance_analytics():

    db = get_db()

    # ---------------------------------------------------
    # GET ALL VALID SESSIONS (ONLY EXISTING USERS)
    # ---------------------------------------------------
    rows = db.execute("""
        SELECT users.name,
               users.role,
               attendance.emp_id,
               attendance.date,
               attendance.checkin_time,
               attendance.checkout_time,
               attendance.worked_hours
        FROM attendance
        INNER JOIN users ON attendance.emp_id = users.id
        WHERE attendance.checkout_time IS NOT NULL
        ORDER BY attendance.date DESC, attendance.checkin_time
    """).fetchall()

    # ---------------------------------------------------
    # MONTHLY SUMMARY
    # ---------------------------------------------------
    monthly_summary = db.execute("""
        SELECT strftime('%Y-%m', attendance.date) as month,
               ROUND(SUM(attendance.worked_hours),2)
        FROM attendance
        INNER JOIN users ON attendance.emp_id = users.id
        WHERE attendance.worked_hours IS NOT NULL
        GROUP BY month
        ORDER BY month DESC
    """).fetchall()

    # ---------------------------------------------------
    # HEATMAP DATA (Daily attendance count)
    # ---------------------------------------------------
    heatmap_data = db.execute("""
        SELECT attendance.date,
               COUNT(DISTINCT attendance.emp_id)
        FROM attendance
        INNER JOIN users ON attendance.emp_id = users.id
        GROUP BY attendance.date
        ORDER BY attendance.date DESC
    """).fetchall()

    # ---------------------------------------------------
    # LEADERBOARD (Top Employees)
    # ---------------------------------------------------
    leaderboard_raw = db.execute("""
        SELECT users.name,
               attendance.emp_id,
               ROUND(SUM(attendance.worked_hours),2) as total_hours
        FROM attendance
        INNER JOIN users ON attendance.emp_id = users.id
        WHERE attendance.worked_hours IS NOT NULL
        GROUP BY attendance.emp_id
        ORDER BY total_hours DESC
    """).fetchall()

    # Add ranking number
    leaderboard = []
    rank = 1
    for name, emp_id, total in leaderboard_raw:
        leaderboard.append({
            "rank": rank,
            "name": name,
            "emp_id": emp_id,
            "total_hours": total
        })
        rank += 1

    # ---------------------------------------------------
    # DAILY TOTAL + OVERTIME + PERFORMANCE
    # ---------------------------------------------------
    daily_data = {}

    for name, role, emp_id, date, checkin, checkout, worked in rows:

        if worked is None:
            continue

        key = (emp_id, date)

        if key not in daily_data:
            daily_data[key] = {
                "name": name,
                "role": role,
                "emp_id": emp_id,
                "date": date,
                "total_hours": 0,
                "sessions": []
            }

        daily_data[key]["total_hours"] += worked
        daily_data[key]["sessions"].append({
            "checkin": checkin,
            "checkout": checkout,
            "hours": round(worked, 2)
        })

    HOURLY_RATE = 200  # change if needed

    for key in daily_data:

        total = daily_data[key]["total_hours"]
        overtime = max(0, total - 8)
        overtime_salary = overtime * HOURLY_RATE * 1.5

        # AI Productivity Prediction Logic
        if total >= 9:
            score = "Excellent ⭐⭐⭐"
            prediction = "Highly Productive"
        elif total >= 8:
            score = "Good ⭐⭐"
            prediction = "Consistent Performer"
        elif total >= 6:
            score = "Average ⭐"
            prediction = "Moderate Productivity"
        else:
            score = "Low ⚠"
            prediction = "Needs Improvement"

        daily_data[key]["total_hours"] = round(total, 2)
        daily_data[key]["overtime"] = round(overtime, 2)
        daily_data[key]["overtime_salary"] = round(overtime_salary, 2)
        daily_data[key]["score"] = score
        daily_data[key]["prediction"] = prediction

    db.close()

    # Convert to sorted list (latest first)
    analytics_data = sorted(
        daily_data.values(),
        key=lambda x: x["date"],
        reverse=True
    )

    return render_template(
        "attendance_analytics.html",
        data=analytics_data,
        monthly_summary=monthly_summary,
        heatmap_data=heatmap_data,
        leaderboard=leaderboard
    )


@app.route("/download/pdf")
def download_pdf():
    user_id_filter = request.args.get("user_id", "").strip()

    db = get_db()
    
    # Fetch monthly summary statistics
    monthly_query = """
        SELECT attendance.emp_id,
               COALESCE(users.name, 'Unknown') AS name,
               users.role,
               strftime('%Y-%m', attendance.date) AS month,
               COUNT(*) AS total_sessions,
               COUNT(DISTINCT attendance.date) AS days_present,
               ROUND(SUM(COALESCE(attendance.worked_hours, 0)), 2) AS total_hours,
               ROUND(AVG(COALESCE(attendance.worked_hours, 0)), 2) AS avg_hours_per_session
        FROM attendance
        LEFT JOIN users
            ON CAST(attendance.emp_id AS INTEGER) = users.id
    """
    
    # Fetch detailed session data (day by day)
    detail_query = """
        SELECT attendance.emp_id,
               COALESCE(users.name, 'Unknown') AS name,
               users.role,
               attendance.date,
               attendance.day,
               attendance.checkin_time,
               attendance.checkout_time,
               ROUND(COALESCE(attendance.worked_hours, 0), 2) AS worked_hours
        FROM attendance
        LEFT JOIN users
            ON CAST(attendance.emp_id AS INTEGER) = users.id
    """

    params = []
    if user_id_filter:
        monthly_query += " WHERE attendance.emp_id = ? "
        detail_query += " WHERE attendance.emp_id = ? "
        params.append(user_id_filter)

    monthly_query += """
        GROUP BY attendance.emp_id, name, users.role, month
        ORDER BY month DESC, name ASC
    """
    
    detail_query += """
        ORDER BY attendance.emp_id, attendance.date ASC, attendance.checkin_time ASC
    """

    monthly_rows = db.execute(monthly_query, tuple(params)).fetchall()
    detail_rows = db.execute(detail_query, tuple(params)).fetchall()
    db.close()

    # Build PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    report_date = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    if user_id_filter:
        title_text = f"<b>Attendance Report - User ID: {user_id_filter}</b>"
    else:
        title_text = "<b>Attendance Report - All Users</b>"
    
    title = Paragraph(title_text, styles['Title'])
    story.append(title)
    story.append(Paragraph(f"<i>Generated on: {report_date}</i>", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    if not monthly_rows:
        story.append(Paragraph("<b>No attendance records found for the selected filter.</b>", styles['Normal']))
    else:
        # ============== MONTHLY SUMMARY SECTION ==============
        story.append(Paragraph("<b><u>Monthly Summary</u></b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        summary_data = [['User ID', 'Name', 'Role', 'Month', 'Days', 'Sessions', 'Total Hrs', 'Avg/Session']]
        for row in monthly_rows:
            summary_data.append([
                str(row[0]),
                row[1],
                row[2] or '-',
                row[3],
                str(row[5]),
                str(row[4]),
                str(row[6]),
                str(row[7])
            ])
        
        summary_table = Table(summary_data, colWidths=[0.7*inch, 1.3*inch, 0.8*inch, 0.8*inch, 0.6*inch, 0.7*inch, 0.8*inch, 0.9*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.4*inch))
        
        # ============== DETAILED DAY-BY-DAY SESSIONS ==============
        story.append(Paragraph("<b><u>Detailed Day-by-Day Sessions</u></b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        # Group by user
        user_sessions = {}
        for row in detail_rows:
            emp_id = row[0]
            if emp_id not in user_sessions:
                user_sessions[emp_id] = {
                    'name': row[1],
                    'role': row[2] or '-',
                    'sessions': []
                }
            user_sessions[emp_id]['sessions'].append({
                'date': row[3],
                'day': row[4],
                'checkin': row[5] or '-',
                'checkout': row[6] or '-',
                'hours': row[7]
            })
        
        # Build tables for each user
        for emp_id in sorted(user_sessions.keys()):
            user_info = user_sessions[emp_id]
            
            # User header
            story.append(Paragraph(
                f"<b>User ID: {emp_id} | Name: {user_info['name']} | Role: {user_info['role']}</b>",
                styles['Heading3']
            ))
            story.append(Spacer(1, 0.05*inch))
            
            # Session table
            session_data = [['Date', 'Day', 'Check-In', 'Check-Out', 'Hours Worked']]
            for session in user_info['sessions']:
                session_data.append([
                    session['date'],
                    session['day'],
                    session['checkin'],
                    session['checkout'],
                    str(session['hours'])
                ])
            
            session_table = Table(session_data, colWidths=[1.2*inch, 1*inch, 1.3*inch, 1.3*inch, 1.2*inch])
            session_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightblue]),
            ]))
            story.append(session_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Footer note
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(
            "<i>Note: All sessions are listed in chronological order. Hours are rounded to 2 decimal places.</i>",
            styles['Normal']
        ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=(
            f"attendance_detailed_report_user_{user_id_filter}.pdf"
            if user_id_filter else "attendance_detailed_report_all_users.pdf"
        )
    )

@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    if request.method == "POST":
        user_id = request.form["user_id"]
        name = request.form["name"]
        role = request.form.get("role", "student")
        
        db = get_db()

        existing_user = db.execute("""
            SELECT id, name FROM users WHERE id = ?
        """, (user_id,)).fetchone()

        training_images = [f for f in os.listdir("TrainingImage") if f.startswith(f"User.{user_id}.")] if os.path.exists("TrainingImage") else []

        # If user exists and has images, they are already fully enrolled.
        if existing_user and training_images:
            db.close()
            return jsonify({"status": "error", "message": f"User ID {user_id} already exists! This user is already enrolled."}), 400

        # If user exists but images are missing, allow retry enrollment with updated profile.
        if existing_user and not training_images:
            db.execute("""
                UPDATE users SET name=?, role=? WHERE id=?
            """, (name, role, user_id))
            db.commit()
            db.close()
            return jsonify({
                "status": "ok",
                "message": "User exists without face data. Ready for face capture retry.",
                "user_id": int(user_id)
            })

        # If user was deleted from DB but files remain, auto-clean orphan images and continue.
        if (not existing_user) and training_images:
            for img in training_images:
                try:
                    os.remove(os.path.join("TrainingImage", img))
                except Exception:
                    pass
        
        # Insert new user
        db.execute("""
            INSERT INTO users (id, name, role)
            VALUES (?, ?, ?)
        """, (user_id, name, role))
        db.commit()
        db.close()

        # Return JSON for AJAX workflow; client triggers /capture separately
        return jsonify({
            "status": "ok",
            "message": "User created. Ready for face capture.",
            "user_id": int(user_id)
        })

    return render_template("enroll.html")

@app.route("/check_user/<int:user_id>")
def check_user(user_id):
    """Check if user already exists"""
    db = get_db()
    existing_user = db.execute("""
        SELECT id, name FROM users WHERE id = ?
    """, (user_id,)).fetchone()
    db.close()
    
    # Check if training images exist
    training_images = [f for f in os.listdir("TrainingImage") if f.startswith(f"User.{user_id}.")] if os.path.exists("TrainingImage") else []
    
    if existing_user:
        return jsonify({"exists": True, "user_name": existing_user[1], "orphan_images": False})

    if training_images:
        # Orphan images (DB row deleted manually) should not block enrollment.
        return jsonify({"exists": False, "orphan_images": True})

    return jsonify({"exists": False, "orphan_images": False})

@app.route("/capture/<int:user_id>")
def capture_faces(user_id):
    global recognizer, camera

    # Release any existing camera connection
    if camera is not None:
        try:
            camera.release()
            camera = None
            time.sleep(0.5)  # Give camera time to release
        except:
            pass

    # Try to open camera with retries.
    # CAP_DSHOW only works on Windows; on Linux/macOS it can fail and prevent enrollment.
    cam = None
    if os.name == "nt":
        backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backend_candidates = [cv2.CAP_ANY]

    for backend in backend_candidates:
        for attempt in range(3):
            cam = cv2.VideoCapture(0, backend)
            if cam.isOpened():
                break

            if cam is not None:
                cam.release()
            time.sleep(0.3)

        if cam is not None and cam.isOpened():
            break
    
    if cam is None or not cam.isOpened():
        print("❌ ERROR: Could not access camera")
        return jsonify({"status": "error", "trained": False, "message": "Camera access failed. Close other apps using camera."}), 500

    # Optimize camera settings for faster capture
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Warm up camera with a few reads
    for _ in range(5):
        cam.read()

    # Load face detector and verify it exists
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_detector.empty():
        cam.release()
        print("❌ ERROR: Haar Cascade file not found or invalid")
        return jsonify({"status": "error", "trained": False, "message": "Face detection model not found. Check haarcascade file."}), 500

    count = 0
    captured_faces = []
    os.makedirs("TrainingImage", exist_ok=True)
    
    TARGET_IMAGES = 12  # Reduced from 20 for speed (still enough for training)
    max_attempts = 450  # Max ~15 seconds at 30fps (gives user enough time to position face)
    attempts = 0
    frame_skip = 0  # Capture every 2nd frame with face for variety
    no_face_count = 0  # Track frames without face detection

    print(f"[ENROLLMENT] Starting capture for user {user_id}...")
    start_time = time.time()

    while count < TARGET_IMAGES and attempts < max_attempts:
        attempts += 1
        
        ret, img = cam.read()
        if not ret:
            print(f"[ENROLLMENT] Warning: Failed to read frame {attempts}")
            continue

        gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray_raw)
        # Reduced minNeighbors from 5 to 3 for easier face detection
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(80, 80))
        if len(faces) == 0:
            # Fallback pass on raw grayscale for cameras where equalization hurts detection.
            faces = face_detector.detectMultiScale(gray_raw, scaleFactor=1.25, minNeighbors=4, minSize=(60, 60))
        if len(faces) > 1:
            faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)

        if len(faces) > 0:
            no_face_count = 0  # Reset no-face counter
            
            # Skip some frames for image variety (every 2nd detection)
            frame_skip += 1
            if frame_skip % 2 != 0:  # Capture every 2nd face detection
                continue
                
            for (x, y, w, h) in faces:
                if count >= TARGET_IMAGES:
                    break
                    
                count += 1
                face_roi = gray[y:y+h, x:x+w]
                captured_faces.append(face_roi)

                # Save the CROPPED FACE ROI directly
                cv2.imwrite(
                    f"TrainingImage/User.{user_id}.{count}.jpg",
                    face_roi
                )
                print(f"[ENROLLMENT] Captured image {count}/{TARGET_IMAGES}")
                break  # Only capture one face per frame
        else:
            no_face_count += 1
            # Log warning if no face detected for extended period
            if no_face_count % 30 == 0:  # Every 30 frames (~1 second)
                print(f"[ENROLLMENT] Warning: No face detected for {no_face_count} frames. Please face the camera.")

    cam.release()
    elapsed_time = time.time() - start_time
    print(f"[ENROLLMENT] Finished: Captured {count}/{TARGET_IMAGES} images in {elapsed_time:.1f}s over {attempts} frames")
    
    # Check if we got enough images
    if count < 8:
        # Cleanup partial enrollment so user can retry immediately without manual DB edits.
        try:
            if os.path.exists("TrainingImage"):
                for img_file in os.listdir("TrainingImage"):
                    if img_file.startswith(f"User.{user_id}."):
                        os.remove(os.path.join("TrainingImage", img_file))
        except Exception:
            pass

        try:
            db = get_db()
            db.execute("DELETE FROM users WHERE id = ?", (user_id,))
            db.commit()
            db.close()
        except Exception:
            pass

        error_msg = f"Only captured {count}/{TARGET_IMAGES} images. Need at least 8."
        
        if count == 0:
            error_msg += " No face detected. Ensure good lighting, remove glasses/mask, and face the camera directly."
        elif count < 4:
            error_msg += " Very few faces detected. Try better lighting and face the camera straight on."
        else:
            error_msg += " Please ensure good lighting and stay centered in the frame."
            
        print(f"[ENROLLMENT] ❌ {error_msg}")
        return jsonify({
            "status": "error", 
            "trained": False, 
            "message": error_msg
        }), 400

    # Check if captured face matches any existing user (detect fake IDs)
    detected_user_id = None
    detected_user_name = None
    min_conf = float('inf')
    
    try:
        if os.path.exists("TrainingImageLabel/Trainner.yml"):
            # Reload latest model before checking
            recognizer.read("TrainingImageLabel/Trainner.yml")
            # Check captured faces against trained model
            for face_roi in captured_faces:
                # CRITICAL: Resize to match training dimensions
                face_roi_resized = cv2.resize(face_roi, (200, 200))
                pred_user_id, conf = recognizer.predict(face_roi_resized)
                
                # If confidence is below threshold and it's a different user
                if conf < THRESHOLD and pred_user_id != user_id:
                    if conf < min_conf:
                        min_conf = conf
                        detected_user_id = pred_user_id
                        
                        # Get the detected user's name
                        db = get_db()
                        user_info = db.execute("""
                            SELECT name FROM users WHERE id = ?
                        """, (pred_user_id,)).fetchone()
                        db.close()
                        detected_user_name = user_info[0] if user_info else f"User {pred_user_id}"
    except Exception as e:
        print(f"Face matching error: {e}")

    # If a different user's face was detected, return error with warning
    if detected_user_id:
        # Delete the wrongly saved images for this fake ID attempt
        for img_file in os.listdir("TrainingImage"):
            if img_file.startswith(f"User.{user_id}."):
                try:
                    os.remove(f"TrainingImage/{img_file}")
                except:
                    pass
        
        # Remove the incorrectly enrolled user from database
        db = get_db()
        db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        db.commit()
        db.close()
        
        return jsonify({
            "status": "duplicate_face", 
            "message": f"⚠️ ALERT: This face is already enrolled! You are already enrolled with ID: {detected_user_id} ({detected_user_name}). You cannot create another account with the same face.",
            "existing_user_id": detected_user_id,
            "existing_user_name": detected_user_name
        }), 400

    # Automatically train the model after capturing images
    print(f"[TRAINING] Starting training with {count} images...")
    training_start = time.time()
    
    if count == 0:
        return jsonify({"status": "captured", "trained": False, "message": "No faces detected. Please try again."})
    
    success = train_model()
    
    if success:
        total_time = time.time() - start_time
        print(f"✅ Total enrollment time: {total_time:.1f}s (capture: {elapsed_time:.1f}s, training: {total_time - elapsed_time:.1f}s)")
        
        # Reload the recognizer with the newly trained model
        try:
            reload_recognizer_from_disk()
            print("✓ Recognizer reloaded")
        except Exception as e:
            print(f"Warning: Could not reload recognizer: {e}")
        return jsonify({
            "status": "captured", 
            "trained": True, 
            "message": f"✅ Enrollment complete! ({count} images, {total_time:.1f}s)"
        })
    else:
        return jsonify({"status": "captured", "trained": False, "message": "Images captured but training failed. Please try again."})

@app.route("/set_threshold", methods=["POST"])
def set_threshold():
    global THRESHOLD
    if not admin_required():
        return "Unauthorized", 403

    requested = int(request.form["threshold"])
    THRESHOLD = max(35, min(70, requested))
    return redirect("/admin")


@app.route("/set_mask_config", methods=["POST"])
def set_mask_config():
    global MASK_AWARE_ENABLED
    global MASK_THRESHOLD_BOOST
    global MASK_REQUIRED_STABLE_MATCHES
    global MASK_MIN_UPPER_CORR
    global MASK_MIN_UPPER_GAP
    global MASK_MAX_UPPER_LBP_DIST
    global MASK_MIN_UPPER_LBP_MARGIN
    global IRIS_VERIFY_ENABLED
    global IRIS_MIN_SCORE
    global IRIS_MIN_GAP
    global IRIS_MIN_VOTES

    if not admin_required():
        return "Unauthorized", 403

    try:
        MASK_AWARE_ENABLED = request.form.get("mask_enabled") == "on"
        MASK_THRESHOLD_BOOST = max(0, min(20, int(request.form.get("mask_threshold_boost", MASK_THRESHOLD_BOOST))))
        MASK_REQUIRED_STABLE_MATCHES = max(2, min(8, int(request.form.get("mask_stable_matches", MASK_REQUIRED_STABLE_MATCHES))))
        MASK_MIN_UPPER_CORR = max(0.20, min(0.95, float(request.form.get("mask_min_upper_corr", MASK_MIN_UPPER_CORR))))
        MASK_MIN_UPPER_GAP = max(0.0, min(0.20, float(request.form.get("mask_min_upper_gap", MASK_MIN_UPPER_GAP))))
        MASK_MAX_UPPER_LBP_DIST = max(0.20, min(3.50, float(request.form.get("mask_max_upper_lbp", MASK_MAX_UPPER_LBP_DIST))))
        MASK_MIN_UPPER_LBP_MARGIN = max(0.0, min(0.50, float(request.form.get("mask_min_upper_lbp_margin", MASK_MIN_UPPER_LBP_MARGIN))))
        IRIS_VERIFY_ENABLED = request.form.get("iris_enabled") == "on"
        IRIS_MIN_SCORE = max(-0.20, min(1.00, float(request.form.get("iris_min_score", IRIS_MIN_SCORE))))
        IRIS_MIN_GAP = max(0.0, min(0.30, float(request.form.get("iris_min_gap", IRIS_MIN_GAP))))
        IRIS_MIN_VOTES = max(1, min(2, int(request.form.get("iris_min_votes", IRIS_MIN_VOTES))))
    except ValueError:
        return redirect("/admin?msg=mask_invalid")

    return redirect("/admin?msg=mask_saved")

@app.route("/stats")
def stats():
    db = get_db()
    data = db.execute("""
    SELECT COUNT(DISTINCT emp_id)
    GROUP BY date
    """).fetchall()
    db.close()
    return jsonify(data)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # 🔐 ADMIN CREDENTIALS (ACADEMIC DEMO)
        if username == "admin" and password == "admin123":
            session["admin"] = True
            return redirect(url_for("admin"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/admin")
def admin():
    if not session.get("admin"):
        return redirect(url_for("login"))   # 🔒 FORCE LOGIN

    db = get_db()
    total_users = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_attendance = db.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    db.close()

    return render_template(
        "admin.html",
        total_users=total_users,
        total_attendance=total_attendance,
        threshold=THRESHOLD,
        mask_aware_enabled=MASK_AWARE_ENABLED,
        mask_threshold_boost=MASK_THRESHOLD_BOOST,
        mask_required_stable_matches=MASK_REQUIRED_STABLE_MATCHES,
        mask_min_upper_corr=MASK_MIN_UPPER_CORR,
        mask_min_upper_gap=MASK_MIN_UPPER_GAP,
        mask_max_upper_lbp_dist=MASK_MAX_UPPER_LBP_DIST,
        mask_min_upper_lbp_margin=MASK_MIN_UPPER_LBP_MARGIN,
        iris_verify_enabled=IRIS_VERIFY_ENABLED,
        iris_min_score=IRIS_MIN_SCORE,
        iris_min_gap=IRIS_MIN_GAP,
        iris_min_votes=IRIS_MIN_VOTES
    )

@app.route("/admin/complaints")
def admin_complaints():
    if not session.get("admin"):
        return redirect(url_for("login"))

    db = get_db()
    complaints = db.execute("""
        SELECT id, category, description, status, date
        FROM complaints
        ORDER BY date DESC
    """).fetchall()
    db.close()

    return render_template("admin_complaints.html", complaints=complaints)


@app.route("/admin/close_complaint/<int:cid>")
def close_complaint(cid):
    if not admin_required():
        return redirect("/login")

    db = get_db()
    db.execute("UPDATE complaints SET status='Closed' WHERE id=?", (cid,))
    db.commit()
    db.close()

    return redirect("/admin/complaints")

@app.route("/admin/system_status")
def system_status():
    if not admin_required():
        return redirect("/login")

    avg_conf = 0
    if confidence_scores:
        avg_conf = round(sum(confidence_scores) / len(confidence_scores), 2)

    return jsonify({
        "camera_status": camera_status,
        "fps": fps_value,
        "success": success_recognition,
        "failure": failed_recognition,
        "avg_confidence": avg_conf
    })

@app.route("/stats/daily")
def daily_stats():
    db = get_db()
    data = db.execute("""
        SELECT date, COUNT(*) 
        FROM attendance
        GROUP BY date
        ORDER BY date
    """).fetchall()
    db.close()
    return jsonify(data)


@app.route("/stats/performance")
def performance_stats():
    return jsonify({
        "success": success_recognition,
        "failure": failed_recognition
    })

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        rating = request.form["rating"]
        message = request.form["message"]

        db = get_db()
        db.execute("""
            INSERT INTO feedback (rating, message, date)
            VALUES (?, ?, date('now'))
        """, (rating, message))
        db.commit()
        db.close()

    return render_template("feedback.html")

@app.route("/admin/feedback")
def admin_feedback():
    if not session.get("admin"):
        return redirect(url_for("login"))

    db = get_db()
    feedback = db.execute("""
        SELECT rating, message, date
        FROM feedback
        ORDER BY date DESC
    """).fetchall()
    db.close()

    return render_template("admin_feedback.html", feedback=feedback)



@app.route("/grievance", methods=["GET", "POST"])
def grievance():
    if request.method == "POST":
        category = request.form["category"]
        description = request.form["description"]

        db = get_db()
        db.execute("""
            INSERT INTO complaints (category, description, date)
            VALUES (?, ?, date('now'))
        """, (category, description))
        db.commit()
        db.close()

    return render_template("grievance.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/camera/on")
def camera_on():
    global camera_enabled
    camera_enabled = True
    return jsonify({"status": "ON"})

@app.route("/camera/off")
def camera_off():
    global camera_enabled, camera
    camera_enabled = False

    if camera is not None:
        camera.release()
        camera = None

    return jsonify({"status": "OFF"})

@app.route("/test_preview")
def test_preview():
    cam = cv2.VideoCapture(0)
    success, frame = cam.read()
    cam.release()

    if not success:
        return jsonify({"status": "fail"})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    result = "No face detected"

    for (x,y,w,h) in faces:
        # Reload latest model before recognition
        if os.path.exists("TrainingImageLabel/Trainner.yml"):
            recognizer.read("TrainingImageLabel/Trainner.yml")
        # CRITICAL: Resize face to match training dimensions
        face_roi_resized = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        uid, conf = recognizer.predict(face_roi_resized)
        if conf < THRESHOLD:
            result = f"Recognized: User {uid}"
        else:
            result = "Face detected but not recognized"

    return jsonify({"status": "ok", "result": result})

@app.route("/admin/model_accuracy")
def model_accuracy():
    total = success_recognition + failed_recognition
    accuracy = round((success_recognition / total) * 100, 2) if total else 0

    return jsonify({
        "accuracy": accuracy,
        "success": success_recognition,
        "failure": failed_recognition
    })


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/admin/accuracy_data")
def accuracy_data():

    date_filter = request.args.get("date")

    db = get_db()
    cur = db.cursor()

    if date_filter:
        cur.execute("""
            SELECT strftime('%H:00', date || ' ' || checkin_time) as hour,
                   SUM(CASE WHEN status != 'failed' THEN 1 ELSE 0 END) as success,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failure
            FROM attendance
            WHERE date = ?
            GROUP BY hour
            ORDER BY hour
        """, (date_filter,))
    else:
        cur.execute("""
            SELECT strftime('%H:00', date || ' ' || checkin_time) as hour,
                   SUM(CASE WHEN status != 'failed' THEN 1 ELSE 0 END) as success,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failure
            FROM attendance
            GROUP BY hour
            ORDER BY hour
        """)

    rows = cur.fetchall()
    db.close()

    result = []

    for hour, success, failure in rows:
        total = (success or 0) + (failure or 0)

        accuracy = round((success / total) * 100, 2) if total else 0

        result.append({
            "hour": hour,
            "accuracy": accuracy
        })

    return jsonify(result)
@app.route("/admin/success_failure_data")
def success_failure_data():
    db = get_db()
    data = db.execute("""
        SELECT date,
               SUM(CASE WHEN status!='failed' THEN 1 ELSE 0 END),
               SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END)
        FROM attendance
        GROUP BY date
    """).fetchall()
    db.close()

    return jsonify({
        "labels": [d[0] for d in data],
        "success": [d[1] for d in data],
        "failure": [d[2] for d in data]
    })
@app.route("/admin/confidence_distribution")
def confidence_distribution():

    # Example ranges
    labels = ["0-50", "50-70", "70-85", "85-100"]
    values = [5, 10, 25, 40]  # Replace with real stored confidence

    return jsonify({
        "labels": labels,
        "values": values
    })



@app.route("/admin/manual_attendance", methods=["POST"])
def manual_attendance():
    emp_id = request.form.get("emp_id")
    date = request.form.get("date")
    checkin = request.form.get("checkin")

    db = get_db()
    db.execute("""
        INSERT INTO attendance (emp_id, date, checkin_time)
        VALUES (?, ?, ?)
    """, (emp_id, date, checkin))

    db.commit()
    db.close()
    return redirect("/attendance")

@app.route("/attendance_status")
def attendance_status_api():
    global attendance_status
    status = attendance_status
    attendance_status = ""  # reset after reading
    return jsonify({"status": status})

@app.route("/support", methods=["GET", "POST"])
def support():

    db = get_db()

    if request.method == "POST":

        form_type = request.form.get("type")
        message = request.form.get("message")
        rating = request.form.get("rating")
        category = request.form.get("category")

        if form_type == "feedback":
            db.execute("""
                INSERT INTO feedback (rating, message, date)
                VALUES (?, ?, date('now'))
            """, (rating, message))

        elif form_type == "grievance":
            db.execute("""
                INSERT INTO complaints (category, description, date)
                VALUES (?, ?, date('now'))
            """, (category, message))

        db.commit()

    db.close()

    return render_template("support.html")

@app.route("/set_mode/<mode>")
def set_mode(mode):
    global attendance_mode

    if mode in ["checkin", "checkout"]:
        attendance_mode = mode

    return jsonify({"mode": attendance_mode})


@app.route("/copilot")
def copilot():
    return render_template("copilot.html")
@app.route("/ai_assistant", methods=["POST"])
def ai_assistant():

    user_msg = request.json.get("message").lower()

    if "check in" in user_msg:
        reply = "To check in, open Live → Check-In Session and complete face verification."

    elif "check out" in user_msg:
        reply = "To check out, open Live → Check-Out Session and complete verification."

    elif "attendance percentage" in user_msg:
        reply = "Attendance percentage = (Today's Attendance / Total Users) × 100."

    elif "camera not working" in user_msg:
        reply = "Make sure no other application is using the camera and refresh the page."

    elif "delete record" in user_msg:
        reply = "Go to Attendance page and use the delete option next to the record."

    else:
        reply = "I'm FAS Copilot 🤖. I can help with attendance, admin tools, analytics, reports, and troubleshooting."

    return jsonify({"reply": reply})
@app.route("/api/copilot", methods=["POST"])
def copilot_api():
    data = request.json
    message = data.get("message", "").lower()

    reply = "I'm FAS Copilot 🤖. How can I help you?"

    # Attendance help
    if "attendance" in message:
        reply = "You can view attendance in the Attendance page. Use filters for daily, weekly, or monthly insights."

    elif "check in" in message:
        reply = "To check-in, select Check-In Session under Live and start the camera."

    elif "check out" in message:
        reply = "To check-out, select Check-Out Session under Live and verify your face."

    elif "admin" in message:
        reply = "Admin dashboard provides analytics, reports, complaints management, and accuracy monitoring."

    elif "accuracy" in message:
        reply = "Model accuracy is calculated using successful recognitions divided by total recognitions."

    elif "report" in message:
        reply = "You can download PDF reports from Attendance or Admin dashboard."

    elif "support" in message or "complaint" in message:
        reply = "Use Support Center to submit complaints or feedback."

    elif "user" in message:
        reply = "You can enroll new users from the Enroll page."

    return jsonify({"reply": reply})

@app.route("/admin/users")
def admin_users():

    if not session.get("admin"):
        return redirect(url_for("login"))

    db = get_db()
    users = db.execute("""
        SELECT id, name, role
        FROM users
        ORDER BY id
    """).fetchall()
    db.close()

    msg = request.args.get("msg", "")
    return render_template("admin_users.html", users=users, msg=msg)
@app.route("/admin/edit_user/<int:user_id>", methods=["POST"])
def edit_user(user_id):

    if not session.get("admin"):
        return redirect(url_for("login"))

    name = request.form.get("name")
    role = request.form.get("role")

    db = get_db()
    db.execute("""
        UPDATE users
        SET name=?, role=?
        WHERE id=?
    """, (name, role, user_id))
    db.commit()
    db.close()

    return redirect("/admin/users")
@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
def delete_user(user_id):

    if not session.get("admin"):
        return redirect(url_for("login"))

    db = get_db()

    # Prevent deleting main admin
    if user_id == 1:
        db.close()
        return redirect("/admin/users?msg=protected")

    # Delete attendance
    db.execute("DELETE FROM attendance WHERE emp_id=?", (user_id,))

    # Delete user
    cur = db.execute("DELETE FROM users WHERE id=?", (user_id,))
    deleted_rows = cur.rowcount

    db.commit()
    db.close()

    # Delete training images
    import os
    folder = "TrainingImage"

    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.startswith(f"User.{user_id}."):
                os.remove(os.path.join(folder, file))

    # Retrain model safely
    if os.path.exists("TrainingImageLabel/Trainner.yml"):
        try:
            train_model()
        except Exception:
            pass

    if deleted_rows and deleted_rows > 0:
        return redirect("/admin/users?msg=deleted")

    return redirect("/admin/users?msg=not_found")

# =============== DIAGNOSTIC ENDPOINTS ===============
@app.route("/debug/training_status")
def training_status():
    """Check current training status and model with detailed diagnostics"""
    try:
        status = {
            "training_images": {},
            "database_users": [],
            "model_exists": os.path.exists("TrainingImageLabel/Trainner.yml"),
            "diagnostics": [],
            "model_info": {}
        }
        
        # Check training images
        if os.path.exists("TrainingImage"):
            images = os.listdir("TrainingImage")
            for img in images:
                if img.endswith(".jpg"):
                    parts = img.split(".")
                    if len(parts) >= 2:
                        user_id = parts[1]
                        if user_id not in status["training_images"]:
                            status["training_images"][user_id] = []
                        status["training_images"][user_id].append(img)
            
            # Convert to count
            training_counts = {uid: len(images) for uid, images in status["training_images"].items()}
            status["training_images"] = training_counts
        
        # Check database users
        db = get_db()
        users = db.execute("SELECT id, name, role FROM users ORDER BY id").fetchall()
        status["database_users"] = [{"id": u[0], "name": u[1], "role": u[2]} for u in users]
        db.close()
        
        # Validation checks
        if not status["training_images"]:
            status["diagnostics"].append("❌ No training images found!")
        
        for user_id_str, count in status["training_images"].items():
            user_id = int(user_id_str)
            user_exists = any(u["id"] == user_id for u in status["database_users"])
            
            if not user_exists:
                status["diagnostics"].append(f"⚠️  Training images exist for user {user_id} but NOT in database!")
            elif count < 10:
                status["diagnostics"].append(f"⚠️  User {user_id} has only {count} training images (need 10+)")
            else:
                status["diagnostics"].append(f"✅ User {user_id} has {count} training images")
        
        for user in status["database_users"]:
            user_id_str = str(user["id"])
            if user_id_str not in status["training_images"]:
                status["diagnostics"].append(f"❌ User {user['id']} ({user['name']}) in DB but NO training images!")
        
        # Check model info if it exists
        if status["model_exists"]:
            try:
                test_recognizer = cv2.face.LBPHFaceRecognizer_create()
                test_recognizer.read("TrainingImageLabel/Trainner.yml")
                status["model_info"] = {
                    "status": "✅ Model file can be loaded",
                    "file_size": os.path.getsize("TrainingImageLabel/Trainner.yml"),
                    "last_modified": os.path.getmtime("TrainingImageLabel/Trainner.yml")
                }
            except Exception as e:
                status["model_info"] = {
                    "status": f"❌ Error loading model: {str(e)}"
                }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug/test_model", methods=["POST"])
def test_model():
    """Test the trained model against actual training images"""
    try:
        if not os.path.exists("TrainingImageLabel/Trainner.yml"):
            return jsonify({"status": "error", "message": "No model file found. Train the model first."}), 400
        
        # Load the recognizer
        test_recognizer = cv2.face.LBPHFaceRecognizer_create()
        test_recognizer.read("TrainingImageLabel/Trainner.yml")
        
        results = {
            "status": "ok",
            "tests": [],
            "accuracy": 0
        }
        
        if not os.path.exists("TrainingImage"):
            return jsonify({"status": "error", "message": "No training images found"}), 400
        
        all_images = [f for f in os.listdir("TrainingImage") if f.endswith(".jpg")]
        correct_predictions = 0
        
        # Test first 3 images from each user
        user_images = {}
        for img in all_images:
            parts = img.split(".")
            if len(parts) >= 2:
                user_id_str = parts[1]
                if user_id_str not in user_images:
                    user_images[user_id_str] = []
                user_images[user_id_str].append(img)
        
        for user_id_str in sorted(user_images.keys()):
            actual_user_id = int(user_id_str)
            images = user_images[user_id_str][:3]  # Test first 3 images
            
            for img_name in images:
                try:
                    # Load and prepare image
                    img_path = os.path.join("TrainingImage", img_name)
                    gray_img = Image.open(img_path).convert("L")
                    img_np = np.array(gray_img, "uint8")
                    img_resized = cv2.resize(img_np, (200, 200))
                    
                    # Predict
                    predicted_id, conf = test_recognizer.predict(img_resized)
                    
                    is_correct = (predicted_id == actual_user_id)
                    if is_correct:
                        correct_predictions += 1
                    
                    results["tests"].append({
                        "image": img_name,
                        "actual_user_id": actual_user_id,
                        "predicted_user_id": predicted_id,
                        "confidence": round(conf, 2),
                        "correct": is_correct,
                        "status": "✅ MATCH" if is_correct else "❌ MISMATCH"
                    })
                except Exception as e:
                    results["tests"].append({
                        "image": img_name,
                        "error": str(e)
                    })
        
        if results["tests"]:
            results["accuracy"] = round((correct_predictions / len(results["tests"])) * 100, 2)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/debug/retrain", methods=["POST"])
def retrain_model():
    """Force retrain the model with maximum diagnostics"""
    try:
        # Delete old model to ensure clean training
        if os.path.exists("TrainingImageLabel/Trainner.yml"):
            os.remove("TrainingImageLabel/Trainner.yml")
            print("[RETRAIN] Deleted old model file")
        
        print("[RETRAIN] Starting model training from scratch...")
        result = train_model()
        
        if not result:
            return jsonify({
                "status": "error", 
                "message": "❌ Training failed - check Flask console for details"
            }), 400
        
        # Verify model was created
        if not os.path.exists("TrainingImageLabel/Trainner.yml"):
            return jsonify({
                "status": "error",
                "message": "❌ Model file not created after training"
            }), 400
        
        # Reload the global recognizer
        global recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainner.yml")
        print("[RETRAIN] ✅ Global recognizer reloaded with new model")
        
        return jsonify({
            "status": "success",
            "message": "✅ Model retrained and reloaded successfully. Test with /debug/test_model"
        })
    except Exception as e:
        print(f"[ERROR] Retrain failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/debug/attendance_records")
def debug_attendance_records():
    """Check all attendance records in database"""
    try:
        db = get_db()
        
        # Get raw attendance records
        records = db.execute("""
            SELECT id, emp_id, date, day, checkin_time, checkout_time, worked_hours
            FROM attendance
            ORDER BY id DESC
            LIMIT 20
        """).fetchall()
        
        # Get user info
        users = db.execute("SELECT id, name FROM users").fetchall()
        db.close()
        
        user_map = {str(u[0]): u[1] for u in users}
        
        result = {
            "total_records": len(records),
            "recent_records": [],
            "user_map": user_map
        }
        
        for rec in records:
            emp_id_db = rec[1]  # What's stored in database
            
            # Try to get user name from database
            db = get_db()
            user_name_query = db.execute(
                "SELECT name FROM users WHERE id = ?",
                (emp_id_db,)
            ).fetchone()
            db.close()
            
            user_name_db = user_name_query[0] if user_name_query else "NOT FOUND"
            
            result["recent_records"].append({
                "attendance_id": rec[0],
                "emp_id_stored": emp_id_db,
                "user_name_from_db": user_name_db,
                "date": rec[2],
                "checkin_time": rec[4],
                "checkout_time": rec[5],
                "worked_hours": rec[6]
            })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
def clear_all_training():
    """DANGEROUS: Clear ALL training data and start fresh"""
    try:
        import shutil
        
        # Delete training images folder
        if os.path.exists("TrainingImage"):
            shutil.rmtree("TrainingImage")
            print("[CLEANUP] Deleted TrainingImage folder")
        
        # Delete model folder
        if os.path.exists("TrainingImageLabel"):
            shutil.rmtree("TrainingImageLabel")
            print("[CLEANUP] Deleted TrainingImageLabel folder")
        
        # Delete all users from database
        db = get_db()
        db.execute("DELETE FROM users")
        db.execute("DELETE FROM attendance")
        db.commit()
        db.close()
        print("[CLEANUP] Cleared all users and attendance records from database")
        
        return jsonify({
            "status": "success",
            "message": "✅ All training data, users, and attendance records have been cleared. You can now start fresh with enrollment."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ------------------ Main ------------------
if __name__ == "__main__":
    app.run(debug=True)

