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
camera_enabled = False




from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import session



# ------------------ Flask App ------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
# ------------------ Config ------------------
CAMERA_INDEX = 0
THRESHOLD = 65          # LBPH threshold (lower = stricter)
TOTAL_EMPLOYEES = 10    # change later (will auto-calc with users table)

# ------------------ OpenCV Setup ------------------
camera = cv2.VideoCapture(CAMERA_INDEX)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("TrainingImageLabel/Trainner.yml")

# ------------------ Database ------------------
def get_db():
    return sqlite3.connect("database/attendance.db")

# ------------------ Attendance Logic ------------------
def mark_attendance(user_id):
    today = datetime.date.today().isoformat()
    time_now = datetime.datetime.now().strftime("%H:%M:%S")

    db = get_db()
    cur = db.cursor()

    cur.execute("""
        INSERT OR IGNORE INTO attendance (user_id, date, time)
        VALUES (?, ?, ?)
    """, (user_id, today, time_now))

    db.commit()
    db.close()

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
    global blink_detected, head_moved, prev_face_x, camera_enabled

    while True:
        if not camera_enabled:
            time.sleep(0.2)
            continue
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        status_text = "Liveness: Not Verified"
        status_color = (0, 0, 255)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            # ---------- Blink Detection ----------
            eyes = eye_cascade.detectMultiScale(face_roi)
            if len(eyes) == 0:
                blink_detected = True

            # ---------- Head Movement ----------
            face_center_x = x + w // 2
            if prev_face_x is not None:
                if abs(face_center_x - prev_face_x) > 20:
                    head_moved = True
            prev_face_x = face_center_x

            # ---------- Liveness Decision ----------
            if blink_detected and head_moved:
                status_text = "Liveness: Verified"
                status_color = (0, 255, 0)

                user_id, conf = recognizer.predict(face_roi)
                if conf < THRESHOLD:
                    label = f"ID {user_id}"
                    color = (0, 255, 0)
                    mark_attendance(user_id)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
            else:
                label = "Complete Liveness Check"
                color = (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(frame, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


        
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    faces = []
    ids = []

    image_paths = [os.path.join("TrainingImage", f)
                   for f in os.listdir("TrainingImage")]

    for image_path in image_paths:
        gray_img = Image.open(image_path).convert("L")
        img_np = np.array(gray_img, "uint8")

        user_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces_detected = detector.detectMultiScale(img_np)

        for (x, y, w, h) in faces_detected:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(user_id)

    recognizer.train(faces, np.array(ids))
    os.makedirs("TrainingImageLabel", exist_ok=True)
    recognizer.save("TrainingImageLabel/Trainner.yml")

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

@app.route("/attendance")
def attendance():
    db = get_db()
    rows = db.execute("""
SELECT attendance.id, users.name, attendance.date, attendance.time
FROM attendance
JOIN users ON users.id = attendance.user_id
ORDER BY attendance.date DESC
""").fetchall()

    db.close()

    today_count, total_count, percentage = get_attendance_stats()

    return render_template(
        "attendance.html",
        rows=rows,
        today_count=today_count,
        total_count=total_count,
        percentage=percentage
    )

@app.route("/download/pdf")
def download_pdf():
    db = get_db()
    rows = db.execute("""
        SELECT user_id, date, time
        FROM attendance
        ORDER BY date DESC, time DESC
    """).fetchall()
    db.close()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    text = pdf.beginText(40, 750)
    text.setFont("Helvetica", 10)

    text.textLine("Attendance Report")
    text.textLine("-------------------------------")

    for r in rows:
        text.textLine(f"User ID: {r[0]} | Date: {r[1]} | Time: {r[2]}")

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
        # Remove email and department if table doesn't have these columns
        
        db = get_db()
        # Match the existing table structure (id, name, role)
        db.execute("""
            INSERT OR REPLACE INTO users (id, name, role)
            VALUES (?, ?, 'user')
        """, (user_id, name))
        db.commit()
        db.close()

        return redirect(url_for("capture_faces", user_id=user_id))

    return render_template("enroll.html")

@app.route("/capture/<int:user_id>")
def capture_faces(user_id):
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    count = 0
    os.makedirs("TrainingImage", exist_ok=True)

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(
                f"TrainingImage/User.{user_id}.{count}.jpg",
                gray[y:y+h, x:x+w]
            )

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow("Enrollment - Press Q to stop", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()

    return jsonify({"status": "captured"})


@app.route("/train")
def train():
    train_model()
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
    SELECT date, COUNT(*) FROM attendance
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
    global camera_enabled
    camera_enabled = False
    return jsonify({"status": "OFF"})

@app.route("/train_and_test")
def train_and_test():
    try:
        train_model()

        # simple test: check if model file exists
        if os.path.exists("TrainingImageLabel/Trainner.yml"):
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "fail"})

    except Exception as e:
        return jsonify({"status": "fail", "error": str(e)})
    
@app.route("/train_test_progress")
def train_test_progress():
    try:
        # STEP 1: Train
        train_model()

        # STEP 2: Simple test (load model)
        test_recognizer = cv2.face.LBPHFaceRecognizer_create()
        test_recognizer.read("TrainingImageLabel/Trainner.yml")

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
        uid, conf = recognizer.predict(gray[y:y+h, x:x+w])
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
    total = success_recognition + failed_recognition
    acc = round((success_recognition / total) * 100, 2) if total else 0

    return jsonify({
        "labels": ["Accuracy"],
        "values": [acc]
    })




# ------------------ Main ------------------
if __name__ == "__main__":
    app.run(debug=True)
