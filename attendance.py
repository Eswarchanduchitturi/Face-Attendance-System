import cv2
import numpy as np
import pandas as pd
import os
import time
import datetime

# ------------------ Load Trained Model ------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("TrainingImageLabel/Trainner.yml")

# ------------------ Load Haar Cascade -------------------
harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath)

# ------------------ Load Employee Details ----------------
df = pd.read_csv("EmployeeDetails" + os.sep + "EmployeeDetails.csv")

font = cv2.FONT_HERSHEY_SIMPLEX
col_names = ['Id', 'Name', 'Date', 'Time']
attendance = pd.DataFrame(columns=col_names)

# ------------------ Start Camera ------------------------
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# ------------------ Face Recognition Loop ----------------
while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        confidence = int(100 - conf)

        if confidence > 50:
            name = df.loc[df['Id'] == Id]['Name'].values
            name = str(name)[2:-2]

            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

            attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
            label = f"{Id}-{name}"
        else:
            label = "Unknown"

        cv2.putText(frame, label, (x, y-10), font, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"{confidence}%", (x, y+h+20), font, 0.8, (0, 255, 0), 2)

    attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------ Save Attendance ----------------------
ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H-%M-%S')

fileName = "Attendance" + os.sep + f"Attendance_{date}_{timeStamp}.csv"
attendance.to_csv(fileName, index=False)

print("✅ Attendance Recorded Successfully")

cam.release()
cv2.destroyAllWindows()