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

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("TrainingImageLabel/Trainner.yml")

# ------------------ Database ------------------
def get_db():
    return sqlite3.connect("database/attendance.db")

# ------------------ Attendance Logic ------------------
def mark_attendance(user_id):
    global attendance_mode

    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")
    day = now.strftime("%A")
    time_now = now.strftime("%H:%M:%S")

    db = get_db()
    cur = db.cursor()

    # Get today's record
    cur.execute("""
        SELECT checkin_time, checkout_time
        FROM attendance
        WHERE emp_id=? AND date=?
    """, (user_id, today))

    record = cur.fetchone()

    # -------------------------
    # CHECK-IN MODE
    # -------------------------
    if attendance_mode == "checkin":

        if record is None:
            cur.execute("""
                INSERT INTO attendance (emp_id, date, day, checkin_time)
                VALUES (?, ?, ?, ?)
            """, (user_id, today, day, time_now))

            db.commit()
            db.close()
            return "checkin_success"

        db.close()
        return "already_checked_in"

    # -------------------------
    # CHECK-OUT MODE
    # -------------------------
    elif attendance_mode == "checkout":

        if record and record[1] is None:

            login_dt = datetime.datetime.strptime(record[0], "%H:%M:%S")
            logout_dt = datetime.datetime.strptime(time_now, "%H:%M:%S")

            worked_hours = (logout_dt - login_dt).total_seconds() / 3600

            cur.execute("""
                UPDATE attendance
                SET checkout_time=?, worked_hours=?
                WHERE emp_id=? AND date=?
            """, (time_now, worked_hours, user_id, today))

            db.commit()
            db.close()
            return f"checkout_success_{round(worked_hours,2)}"

        db.close()
        return "already_checked_out"



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
    global attendance_status, success_recognition, failed_recognition, camera

    camera = cv2.VideoCapture(CAMERA_INDEX)

    while True:

        if not camera_enabled:
            break

        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        status_text = "Detecting Face..."
        status_color = (0, 200, 255)

        if len(faces) > 0:
            status_text = "Face Detected"
            status_color = (255, 200, 0)

        for (x, y, w, h) in faces:

            face_roi = gray[y:y+h, x:x+w]

            user_id, conf = recognizer.predict(face_roi)

            if conf < THRESHOLD:

                status_text = "Identity Verified"
                status_color = (0, 255, 0)

                result = mark_attendance(user_id)

                attendance_status = result

                success_recognition += 1

                # Prevent rapid repeated marking
                import time
                time.sleep(2)

            else:
                status_text = "Unknown Face"
                status_color = (0, 0, 255)

                failed_recognition += 1
                attendance_status = "failed"

            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)

        # 🔥 Draw background for status text (OUTSIDE face loop)
        cv2.rectangle(frame, (10, 10), (700, 80), (0, 0, 0), -1)

        cv2.putText(frame,
                    status_text,
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    status_color,
                    2)

        _, buffer = cv2.imencode(".jpg", frame)

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    camera.release()


        
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

    # Get all attendance records
    rows = db.execute("""
        SELECT id, emp_id, date, day,
               checkin_time, checkout_time, worked_hours
        FROM attendance
        ORDER BY date DESC
    """).fetchall()

    # Total registered users
    total_users = db.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    # Total attendance sessions
    total_count = db.execute(
        "SELECT COUNT(*) FROM attendance"
    ).fetchone()[0]

    # Today's attendance count
    today = datetime.date.today().isoformat()

    today_count = db.execute("""
        SELECT COUNT(DISTINCT emp_id)
        FROM attendance
        WHERE date = ?
    """, (today,)).fetchone()[0]

    db.close()

    # Calculate percentage
    percentage = 0
    if total_users > 0:
        percentage = round(
            (today_count / total_users) * 100,
            2
        )

    return render_template(
        "attendance.html",
        rows=rows,
        total_users=total_users,
        total_count=total_count,
        today_count=today_count,
        percentage=percentage
    )


@app.route("/download/pdf")
def download_pdf():
    db = get_db()
    rows = db.execute("""
        SELECT emp_id, date, checkin_time, checkout_time, worked_hours
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
    text.textLine(
    f"User ID: {r[0]} | Date: {r[1]} | "
    f"In: {r[2]} | Out: {r[3]} | "
    f"Hours: {round(r[4],2) if r[4] else 'N/A'}"
)


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
@app.route("/admin/delete_attendance/<int:record_id>")
def delete_attendance(record_id):

    if not session.get("admin"):
        return redirect("/login")

    db = get_db()
    db.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
    db.commit()
    db.close()

    return redirect("/attendance")
@app.route("/admin/delete_employee_attendance/<emp_id>")
def delete_employee_attendance(emp_id):

    if not session.get("admin"):
        return redirect("/login")

    db = get_db()
    db.execute("DELETE FROM attendance WHERE emp_id = ?", (emp_id,))
    db.commit()
    db.close()

    return redirect("/attendance")
@app.route("/admin/edit_attendance/<int:record_id>", methods=["POST"])
def edit_attendance(record_id):

    checkin = request.form.get("checkin_time")
    checkout = request.form.get("checkout_time")

    db = get_db()
    db.execute("""
        UPDATE attendance
        SET checkin_time=?, checkout_time=?
        WHERE id=?
    """, (checkin, checkout, record_id))

    db.commit()
    db.close()

    return redirect("/attendance")
@app.route("/admin/bulk_delete", methods=["POST"])
def bulk_delete():

    ids = request.json.get("ids")

    db = get_db()
    for rid in ids:
        db.execute("DELETE FROM attendance WHERE id=?", (rid,))
    db.commit()
    db.close()

    return jsonify({"status": "success"})

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





# ------------------ Main ------------------
if __name__ == "__main__":
    app.run(debug=True)
