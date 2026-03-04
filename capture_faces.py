import cv2
import os

# Try to open camera with DirectShow backend (more reliable on Windows)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if camera opened successfully
if not cam.isOpened():
    print("❌ ERROR: Could not access camera!")
    print("Make sure:")
    print("  1. No other application is using the camera")
    print("  2. Camera permissions are granted")
    print("  3. Camera is properly connected")
    exit(1)

# Set camera resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if face_detector.empty():
    print("❌ ERROR: Could not load haarcascade_frontalface_default.xml")
    print("Make sure the file exists in the current directory")
    cam.release()
    exit(1)

face_id = 1  # MUST match EmployeeDetails.csv
face_id = 2
count = 0
failed_reads = 0

os.makedirs("TrainingImage", exist_ok=True)

print("✓ Camera initialized successfully")
print("✓ Face detector loaded")
print(f"📸 Starting face capture for User ID: {face_id}")
print("Press ESC to stop or capture 50 images")
print("-" * 50)

while True:
    ret, img = cam.read()
    
    # Check if frame was captured successfully
    if not ret or img is None:
        failed_reads += 1
        print(f"⚠ Warning: Failed to read frame ({failed_reads} failures)")
        
        if failed_reads > 10:
            print("❌ ERROR: Too many failed frame reads. Camera may be disconnected.")
            break
        continue
    
    # Reset failed counter on successful read
    failed_reads = 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"TrainingImage/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(img, f"Captured: {count}/50", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"✓ Captured image {count}/50")

    cv2.imshow('Capturing Faces - Press ESC to stop', img)

    if cv2.waitKey(1) == 27 or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()
print("-" * 50)
print(f"✅ Face capture complete! Collected {count} images")
print(f"📁 Images saved to: TrainingImage/User.{face_id}.*.jpg")
