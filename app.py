from flask import Flask, render_template, Response, request, redirect, url_for, send_file, jsonify, session

import cv2
import sqlite3
import datetime
import os
import io
import math

import numpy as np
from PIL import Image
import time



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
from flask import session



# ------------------ Flask App ------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
# ------------------ Config ------------------
CAMERA_INDEX = 0
THRESHOLD = 65          # LBPH threshold (lower = stricter) - SET FOR BALANCED ACCURACY
TOTAL_EMPLOYEES = 10    # change later (will auto-calc with users table)

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

    if camera is None:
        camera = cv2.VideoCapture(CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while camera_enabled:

        success, frame = camera.read()
        if not success:
            break

        # Mirror camera (front camera style)
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        status_text = "Detecting Face..."
        status_color = (0, 200, 255)
        user_name = ""

        recognized_faces = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Force reload recognizer to get latest trained model
            model_loaded = False
            try:
                if os.path.exists("TrainingImageLabel/Trainner.yml"):
                    recognizer.read("TrainingImageLabel/Trainner.yml")
                    model_loaded = True
            except Exception as e:
                print(f"Warning: Could not reload recognizer model: {e}")
            
            # Skip prediction if model not available
            if not model_loaded:
                box_color = (0, 140, 255)
                label = "No Model - Enroll Users First"
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                continue
            
            # CRITICAL: Resize face to match training image dimensions (200x200)
            face_roi_resized = cv2.resize(face_roi, (200, 200))
            
            user_id, conf = recognizer.predict(face_roi_resized)
            print(f"[RECOGNITION] Predicted user_id: {user_id}, confidence: {conf:.2f}")

            box_color = (0, 0, 255)
            label = f"Unknown ({conf:.1f})"

            if conf < THRESHOLD:
                db = get_db()
                user_check = db.execute("""
                    SELECT id, name FROM users WHERE id = ?
                """, (user_id,)).fetchone()
                db.close()

                if user_check:
                    detected_id = user_check[0]
                    detected_name = user_check[1]
                    box_color = (0, 255, 0)
                    label = f"ID: {detected_id} - {detected_name} ({conf:.1f})"
                    print(f"[MATCH] User {detected_id} ({detected_name}) matched with confidence {conf:.2f}")

                    face_image = frame[y:y+h, x:x+w].copy()
                    recognized_faces.append({
                        "user_id": user_id,
                        "user_name": detected_name,
                        "conf": conf,
                        "face_image": face_image
                    })
                else:
                    label = f"Invalid ID {user_id} ({conf:.1f})"

            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2)

        if len(recognized_faces) == 1:
            best_match = recognized_faces[0]
            user_name = best_match["user_name"]
            status_text = f"Welcome {user_name}!"
            status_color = (0, 255, 0)
            
            print(f"\n[ATTENDANCE_RECORD] Best face match detected:")
            print(f"  User ID: {best_match['user_id']}")
            print(f"  User Name: {user_name}")
            print(f"  Confidence: {best_match['conf']:.2f}")
            print(f"  THRESHOLD: {THRESHOLD}")

            current_time = time.time()
            if current_time - last_attendance_time > ATTENDANCE_COOLDOWN:
                result = mark_attendance(best_match["user_id"], best_match["face_image"])
                print(f"  Attendance recording result: {result}")
                attendance_status = result
                success_recognition += 1
                last_attendance_time = current_time
        elif len(recognized_faces) > 1:
            status_text = "Multiple recognized faces - show one face only"
            status_color = (0, 140, 255)
            attendance_status = "multiple_faces"
        elif len(faces) > 0:
            status_text = "Face not recognized"
            status_color = (0, 0, 255)
            attendance_status = "failed"
            failed_recognition += 1

        # Status text background
        cv2.rectangle(frame, (10, 10), (600, 70), (0, 0, 0), -1)

        cv2.putText(frame,
                    status_text,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    status_color,
                    2)

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
            
            user_face_count[user_id] = user_face_count.get(user_id, 0) + 1
                
        except Exception:
            continue

    if len(faces) == 0:
        print(f"❌ ERROR: No valid faces found for training!")
        return False

    print(f"[TRAINING] Training model with {len(faces)} samples from {len(user_face_count)} users...")

    try:
        recognizer.train(faces, np.array(ids, dtype=np.int32))
        
        os.makedirs("TrainingImageLabel", exist_ok=True)
        recognizer.save("TrainingImageLabel/Trainner.yml")
        
        print(f"✅ Training complete!")
        return True
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        return False

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
    rows = db.execute("""
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

    total_count = len(rows)

    percentage = 0
    if total_users > 0:
        percentage = round((today_count / total_users) * 100, 2)

    db.close()

    return render_template(
        "attendance.html",
        rows=rows,
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
    db = get_db()
    rows = db.execute("""
        SELECT attendance.emp_id,
               COALESCE(users.name, 'Unknown'),
               attendance.date,
               attendance.checkin_time,
               attendance.checkout_time,
               attendance.worked_hours
        FROM attendance
        LEFT JOIN users
            ON CAST(attendance.emp_id AS INTEGER) = users.id
        ORDER BY attendance.date DESC, attendance.checkin_time DESC
    """).fetchall()
    db.close()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    text = pdf.beginText(40, 750)
    text.setFont("Helvetica", 10)

    text.textLine("Attendance Report")
    text.textLine("-------------------------------")
    for r in rows:
        text.textLine(
            f"User ID: {r[0]} | Name: {r[1]} | Date: {r[2]} | "
            f"In: {r[3] or '-'} | Out: {r[4] or '-'} | "
            f"Hours: {round(r[5], 2) if r[5] is not None else 'N/A'}"
        )

    pdf.drawText(text)
    pdf.save()

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="attendance_report.pdf"
    )

@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    if request.method == "POST":
        user_id = request.form["user_id"]
        name = request.form["name"]
        role = request.form.get("role", "student")
        
        db = get_db()
        
        # Check if user already exists
        existing_user = db.execute("""
            SELECT id, name FROM users WHERE id = ?
        """, (user_id,)).fetchone()
        
        if existing_user:
            db.close()
            return jsonify({"status": "error", "message": f"User ID {user_id} already exists! This user is already enrolled."}), 400
        
        # Check if training images already exist for this user
        training_images = [f for f in os.listdir("TrainingImage") if f.startswith(f"User.{user_id}.")] if os.path.exists("TrainingImage") else []
        if training_images:
            db.close()
            return jsonify({"status": "error", "message": f"Training images already exist for user ID {user_id}. This user cannot be re-enrolled."}), 400
        
        # Insert new user
        db.execute("""
            INSERT INTO users (id, name, role)
            VALUES (?, ?, ?)
        """, (user_id, name, role))
        db.commit()
        db.close()

        return redirect(url_for("capture_faces", user_id=user_id))

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
    
    if existing_user or training_images:
        return jsonify({"exists": True, "user_name": existing_user[1] if existing_user else "Unknown"})
    else:
        return jsonify({"exists": False})

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

    # Try to open camera with retries
    cam = None
    for attempt in range(3):
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend on Windows
        if cam.isOpened():
            break
        time.sleep(0.3)
    
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
    max_attempts = 150  # Max 150 frames = 5 seconds at 30fps (gives user time to position face)
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

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Reduced minNeighbors from 5 to 3 for easier face detection
        faces = face_detector.detectMultiScale(gray, 1.3, 3)

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
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("TrainingImageLabel/Trainner.yml")
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



@app.route("/train")
def train():
    global recognizer
    success = train_model()
    if success:
        # Reload the recognizer with the newly trained model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainner.yml")
        print("✓ Recognizer reloaded with new model")
    return render_template("train.html")



@app.route("/set_threshold", methods=["POST"])
def set_threshold():
    global THRESHOLD
    if not admin_required():
        return "Unauthorized", 403

    THRESHOLD = int(request.form["threshold"])
    return redirect("/admin")

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
        threshold=THRESHOLD
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

@app.route("/train_and_test")
def train_and_test():
    try:
        # ✅ Ensure training images exist
        if not os.path.exists("TrainingImage"):
            return jsonify({"status": "fail", "error": "No training images found"})

        images = [
            f for f in os.listdir("TrainingImage")
            if f.endswith(".jpg")
        ]

        if len(images) == 0:
            return jsonify({"status": "fail", "error": "No face images available"})

        # ✅ Train model
        train_model()

        # ✅ Verify model file saved (faster than reloading)
        model_path = "TrainingImageLabel/Trainner.yml"

        if os.path.exists(model_path):
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "fail", "error": "Model not saved"})

    except Exception as e:
        return jsonify({
            "status": "fail",
            "error": str(e)
        })

@app.route("/train_test_progress")
def train_test_progress():
    try:
        # STEP 1: Train
        train_model()

        # STEP 2: Simple test (load model)
        # Just confirm file exists
        if not os.path.exists("TrainingImageLabel/Trainner.yml"):
            raise Exception("Model not saved")


        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"status": "fail", "error": str(e)})

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
