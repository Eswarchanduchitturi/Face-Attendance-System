"""
Improved Face Capture Script
Captures face images with better preprocessing and quality control.
Ensures training data is clean and consistent.
"""

import cv2
import os
import sys
from face_utils import load_cascade_classifiers, preprocess_face, is_blurry

# Load cascade classifiers
face_cascade, _ = load_cascade_classifiers()

if face_cascade is None:
    print("❌ ERROR: Could not load face cascade classifier!")
    sys.exit(1)

# Get user ID from command line or use default
if len(sys.argv) > 1:
    try:
        face_id = int(sys.argv[1])
    except ValueError:
        print("❌ Usage: python capture_faces.py <user_id>")
        print("   Example: python capture_faces.py 1")
        sys.exit(1)
else:
    face_id = input("Enter User ID:  ")
    try:
        face_id = int(face_id)
    except ValueError:
        print("❌ User ID must be a number!")
        sys.exit(1)

# Create output directory
os.makedirs("TrainingImage", exist_ok=True)

# Try to open camera with DirectShow backend (Windows) or default
try:
    if os.name == "nt":
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("❌ ERROR: Could not access camera!")
        print("Make sure:")
        print("  1. No other application is using the camera")
        print("  2. Camera permissions are granted")
        print("  3. Camera is properly connected")
        sys.exit(1)
    
    # Set camera resolution
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✅ Camera initialized successfully")
    print("✅ Face cascade detector loaded")
    print(f"📸 Starting face capture for User ID: {face_id}")
    print("\nInstructions:")
    print("  - Position your face in good lighting")
    print("  - Look straight at the camera")
    print("  - Avoid shadows on your face")
    print("  - Keep expression neutral")
    print("\nPress 'SPACE' to capture, 'ESC' to finish")
    print(f"Target: 50+ images | Current: 0/50")
    print("-" * 60)
    
    count = 0
    failed_reads = 0
    skip_count = 0
    max_skip = 30  # Skip frames to avoid too-similar images
    skip_counter = 0
    
    while True:
        ret, frame = cam.read()
        
        # Check if frame was captured successfully
        if not ret or frame is None:
            failed_reads += 1
            if failed_reads > 10:
                print("❌ ERROR: Too many failed frame reads!")
                break
            continue
        
        failed_reads = 0
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        # Draw status
        status_text = "No face detected - position your face in frame"
        status_color = (0, 0, 255)  # Red
        
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Check quality
            if is_blurry(face_roi, threshold=80):
                status_text = f"Blurry image - move slowly ({skip_counter}/{max_skip})"
                status_color = (0, 165, 255)  # Orange
            else:
                status_text = f"Good quality - Press SPACE to capture ({skip_counter}/{max_skip})"
                status_color = (0, 255, 0)  # Green
                skip_counter += 1
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)
            cv2.putText(frame, f"Face size: {w}x{h}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        elif len(faces) > 1:
            status_text = f"Multiple faces detected ({len(faces)}) - show only ONE face"
            status_color = (0, 0, 255)  # Red
            skip_counter = 0
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)
        
        else:
            skip_counter = 0
        
        # Draw status bar
        cv2.rectangle(frame, (0, 0), (640, 60), (30, 30, 30), -1)
        cv2.putText(frame, status_text, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Captured: {count}/50", (450, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Display frame
        cv2.imshow('Face Capture - Press SPACE to capture, ESC to finish', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC - finish
            break
        
        elif key == 32 and len(faces) == 1 and skip_counter >= max_skip:  # SPACE
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Final blurriness check
            if is_blurry(face_roi, threshold=80):
                print(f"⚠️  Skipped - image too blurry")
                skip_counter = 0
                continue
            
            # Preprocess and save
            face_preprocessed = preprocess_face(face_roi)
            if face_preprocessed is not None:
                filename = f"TrainingImage/User.{face_id}.{count+1}.jpg"
                cv2.imwrite(filename, face_preprocessed)
                count += 1
                print(f"✅ Captured image {count}/50 - {filename}")
                skip_counter = 0
    
    cam.release()
    cv2.destroyAllWindows()
    
    print("-" * 60)
    if count > 0:
        print(f"✅ Face capture complete! Collected {count} images")
        print(f"📁 Images saved to: TrainingImage/User.{face_id}.*.jpg")
        if count < 20:
            print("⚠️  Note: Fewer than 20 images. Consider capturing more for better accuracy.")
    else:
        print("❌ No images captured!")

except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)
