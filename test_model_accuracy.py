import cv2
import os
from collections import defaultdict

print("=== TESTING MODEL ACCURACY ===\n")

# Load model
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Test first 3 images from each user
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
training_dir = "TrainingImage"

if not os.path.exists(training_dir):
    print("❌ No training images found")
    exit()

files = os.listdir(training_dir)
user_files = defaultdict(list)

for f in sorted(files):
    if f.startswith("User."):
        parts = f.split(".")
        user_id = parts[1]
        user_files[user_id].append(f)

print("Testing Model Predictions:\n")
total_tests = 0
correct_predictions = 0

for user_id in sorted(user_files.keys()):
    print(f"User {user_id}:")
    for img_file in user_files[user_id][:3]:  # Test first 3 images
        img_path = os.path.join(training_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        # Resize to match training dimensions
        img_resized = cv2.resize(img, (200, 200))
        pred_id, conf = recognizer.predict(img_resized)
        
        is_correct = pred_id == int(user_id)
        symbol = "✅" if is_correct else "❌"
        
        print(f"  {img_file}: Predicted ID {pred_id}, Confidence {conf:.2f} {symbol}")
        
        total_tests += 1
        if is_correct:
            correct_predictions += 1

accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
print(f"\nAccuracy: {correct_predictions}/{total_tests} = {accuracy:.1f}%")

if accuracy < 80:
    print("\n⚠️  Model accuracy is low - face recognition may fail!")
    print("Solution: Re-enroll with more varied angles and lighting")
elif accuracy == 100:
    print("\n✅ Model looks good!")
else:
    print("\n✅ Model accuracy acceptable")
