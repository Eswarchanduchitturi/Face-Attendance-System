# attendance.py

import cv2
import numpy as np
import pandas as pd
import os
import sys
import sqlite3
import datetime

# ------------------ Mode Selection ------------------
if len(sys.argv) < 2:
    print("Usage: python attendance.py [checkin/checkout]")
    sys.exit()

MODE = sys.argv[1].lower()

if MODE not in ["checkin", "checkout"]:
    print("Invalid mode. Use 'checkin' or 'checkout'")
    sys.exit()

# ------------------ Load Trained Model ------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("TrainingImageLabel/Trainner.yml")

# ------------------ Load Haar Cascade -------------------
harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath)

# ------------------ Load Employee Details ----------------
df = pd.read_csv("EmployeeDetails" + os.sep + "EmployeeDetails.csv")

font = cv2.FONT_HERSHEY_SIMPLEX

# ------------------ Database Connection ------------------
def mark_checkin(emp_id):
    conn = sqlite3.connect("database/attendance.db")
    cursor = conn.cursor()

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    day = now.strftime("%A")
    timeStamp = now.strftime("%H:%M:%S")

    # Prevent duplicate check-in
    cursor.execute("""
        SELECT * FROM attendance
        WHERE emp_id=? AND date=? AND checkout_time IS NULL
    """, (emp_id, date))

    record = cursor.fetchone()

    if record:
        conn.close()
        return "Already Checked-In Today"

    cursor.execute("""
        INSERT INTO attendance (emp_id, date, day, checkin_time)
        VALUES (?, ?, ?, ?)
    """, (emp_id, date, day, timeStamp))

    conn.commit()
    conn.close()

    return f"Check-In Successful at {timeStamp}"


def mark_checkout(emp_id):
    conn = sqlite3.connect("database/attendance.db")
    cursor = conn.cursor()

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    logout_time = now.strftime("%H:%M:%S")

    cursor.execute("""
        SELECT checkin_time FROM attendance
        WHERE emp_id=? AND date=? AND checkout_time IS NULL
    """, (emp_id, date))

    record = cursor.fetchone()

    if not record:
        conn.close()
        return "No Check-In Found"

    login_time = datetime.datetime.strptime(record[0], "%H:%M:%S")
    logout = datetime.datetime.strptime(logout_time, "%H:%M:%S")
    worked = (logout - login_time).total_seconds() / 3600

    cursor.execute("""
        UPDATE attendance
        SET checkout_time=?, worked_hours=?
        WHERE emp_id=? AND date=? AND checkout_time IS NULL
    """, (logout_time, worked, emp_id, date))

    conn.commit()
    conn.close()

    return f"Check-Out Successful | Worked {round(worked,2)} hrs"


# ------------------ Start Camera ------------------------
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

print("📷 Camera Started... Press Q to quit")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        confidence = int(100 - conf)

        if confidence > 50:
            name = df.loc[df['Id'] == Id]['Name'].values
            name = str(name)[2:-2]

            label = f"{Id}-{name}"

            if MODE == "checkin":
                message = mark_checkin(str(Id))
            else:
                message = mark_checkout(str(Id))

            cv2.putText(frame, message, (20, 40),
                        font, 0.8, (0, 255, 0), 2)

        else:
            label = "Unknown"

        cv2.putText(frame, label, (x, y-10),
                    font, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
