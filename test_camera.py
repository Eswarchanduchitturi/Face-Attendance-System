import cv2
import sys

print("=== Camera Diagnostic Test ===\n")

# Test 1: Try default camera
print("Test 1: Testing camera with default backend...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera opened successfully with default backend")
    ret, frame = cap.read()
    if ret:
        print("✅ Frame capture successful")
        print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("❌ Camera opened but cannot read frames")
    cap.release()
else:
    print("❌ Cannot open camera with default backend")

# Test 2: Try DirectShow backend (Windows)
print("\nTest 2: Testing camera with DirectShow backend...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap.isOpened():
    print("✅ Camera opened successfully with DirectShow")
    ret, frame = cap.read()
    if ret:
        print("✅ Frame capture successful")
        print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("❌ Camera opened but cannot read frames")
    cap.release()
else:
    print("❌ Cannot open camera with DirectShow")

# Test 3: List available cameras
print("\nTest 3: Scanning for available cameras...")
available_cameras = []
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        available_cameras.append(i)
        cap.release()

if available_cameras:
    print(f"✅ Found cameras at indices: {available_cameras}")
else:
    print("❌ No cameras found")

print("\n=== Diagnostic Complete ===")
print("\nIf camera test failed, try these steps:")
print("1. Close any other apps using the camera (Teams, Zoom, Skype, etc.)")
print("2. Check Windows Settings > Privacy > Camera permissions")
print("3. Restart your computer")
print("4. Try unplugging/replugging external webcam if using one")
