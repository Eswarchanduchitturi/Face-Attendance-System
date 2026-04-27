"""
Improved Face Recognition Model Training Script
Trains LBPH (Local Binary Patterns Histograms) recognizer with data validation.
Ensures model quality and provides clear feedback.
"""

import cv2
import numpy as np
import os
import sys
from PIL import Image
from face_utils import preprocess_face, FACE_WIDTH, FACE_HEIGHT

# LBPH recognizer parameters - tuned for reliable face recognition
RECOGNIZER_RADIUS = 1
RECOGNIZER_NEIGHBORS = 8
RECOGNIZER_GRID_X = 8
RECOGNIZER_GRID_Y = 8
RECOGNIZER_THRESHOLD = 70.0  # Default threshold for recognition

# Training data requirements
MIN_IMAGES_PER_USER = 12  # Require full enrollment set for stable ID-name mapping
MIN_USERS = 1  # Minimum number of people to train

def get_valid_user_ids():
    """Get set of valid user IDs from database or users file."""
    import sqlite3
    valid_ids = set()
    
    try:
        conn = sqlite3.connect("database/attendance.db")
        cursor = conn.cursor()
        rows = cursor.execute("SELECT id FROM users").fetchall()
        conn.close()
        valid_ids = {row[0] for row in rows}
        print(f"✅ Found {len(valid_ids)} valid users in database")
    except Exception as e:
        print(f"⚠️ Could not read database: {e}")
        print("   Will accept any user IDs in training images")
    
    return valid_ids

def validate_training_image(image_path):
    """
    Validate that training image is suitable for training.
    Returns: (is_valid, user_id) or (False, None)
    """
    try:
        # Load image
        gray_img = Image.open(image_path).convert('L')
        img_np = np.array(gray_img, 'uint8')
        
        # Check size
        if img_np.size < 1000:
            return False, None
        
        # Parse filename: User.{user_id}.{number}.jpg
        filename = os.path.split(image_path)[-1]
        parts = filename.split(".")
        
        if len(parts) < 4 or parts[0] != "User":
            return False, None
        
        try:
            user_id = int(parts[1])
            return True, user_id
        except ValueError:
            return False, None
    
    except Exception as e:
        print(f"  ⚠️ Error reading {filename}: {e}")
        return False, None

def train_model():
    """Train a new face recognition model from training images."""
    
    print("\n" + "="*70)
    print("FACE RECOGNITION MODEL TRAINING")
    print("="*70)
    
    # Load recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=RECOGNIZER_RADIUS,
        neighbors=RECOGNIZER_NEIGHBORS,
        grid_x=RECOGNIZER_GRID_X,
        grid_y=RECOGNIZER_GRID_Y,
        threshold=RECOGNIZER_THRESHOLD
    )
    
    print(f"\n📦 LBPH Recognizer Configuration:")
    print(f"   Radius: {RECOGNIZER_RADIUS}")
    print(f"   Neighbors: {RECOGNIZER_NEIGHBORS}")
    print(f"   Grid: {RECOGNIZER_GRID_X}x{RECOGNIZER_GRID_Y}")
    print(f"   Default Threshold: {RECOGNIZER_THRESHOLD}")
    
    # Check training image directory
    if not os.path.exists("TrainingImage"):
        print(f"\n❌ ERROR: TrainingImage folder not found!")
        return False
    
    image_files = [f for f in os.listdir("TrainingImage") 
                   if f.lower().endswith(".jpg")]
    
    if len(image_files) == 0:
        print(f"\n❌ ERROR: No training images found in TrainingImage/")
        return False
    
    print(f"\n📊 DATASET ANALYSIS:")
    print(f"   Total image files: {len(image_files)}")
    
    # Get valid user IDs from database
    valid_user_ids = get_valid_user_ids()
    if not valid_user_ids:
        print("   No database validation - will accept all user IDs")
    
    # Process training images
    faces = []
    ids = []
    user_image_count = {}
    skipped_images = []
    
    print(f"\n📸 PROCESSING IMAGES:")
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join("TrainingImage", image_file)
        is_valid, user_id = validate_training_image(image_path)
        
        if not is_valid:
            skipped_images.append((image_file, "Invalid filename format"))
            continue
        
        # Check if user ID is valid
        if valid_user_ids and user_id not in valid_user_ids:
            skipped_images.append((image_file, f"User ID {user_id} not in database"))
            continue
        
        try:
            # Load and process image
            gray_img = Image.open(image_path).convert('L')
            img_np = np.array(gray_img, 'uint8')
            
            # Preprocess
            face_preprocessed = preprocess_face(img_np)
            
            # Resize to standard size
            face_resized = cv2.resize(face_preprocessed, (FACE_WIDTH, FACE_HEIGHT))
            
            # Add original and flipped version for robustness
            faces.append(face_resized)
            ids.append(user_id)
            
            faces.append(cv2.flip(face_resized, 1))  # Horizontal flip
            ids.append(user_id)
            
            user_image_count[user_id] = user_image_count.get(user_id, 0) + 1
            
            if (i + 1) % 20 == 0:
                print(f"   ✓ Processed {i + 1}/{len(image_files)} images...")
        
        except Exception as e:
            skipped_images.append((image_file, str(e)))
            continue
    
    print(f"   ✓ Processed all {len(image_files)} images")
    
    # Analyze dataset
    print(f"\n👥 USER STATISTICS:")
    
    if not user_image_count:
        print(f"\n❌ ERROR: No valid training images found!")
        return False
    
    total_unique_users = len(user_image_count)
    total_samples = sum(user_image_count.values())
    
    print(f"   Total unique users: {total_unique_users}")
    print(f"   Total preprocessed samples (with flips): {len(faces)}")
    print(f"   Samples per user: min={min(user_image_count.values())}, "
          f"max={max(user_image_count.values())}, "
          f"avg={total_samples/total_unique_users:.1f}")
    
    # Filter by minimum requirement
    eligible_users = {uid: cnt for uid, cnt in user_image_count.items() 
                     if cnt >= MIN_IMAGES_PER_USER}
    
    if not eligible_users:
        print(f"\n❌ ERROR: No users have enough training samples!")
        print(f"   Requirement: minimum {MIN_IMAGES_PER_USER} images per user")
        print(f"   Available:")
        for uid, cnt in sorted(user_image_count.items()):
            print(f"      User {uid}: {cnt} images")
        return False
    
    # Filter training data
    filtered_faces = []
    filtered_ids = []
    
    for face_img, uid in zip(faces, ids):
        if uid in eligible_users:
            filtered_faces.append(face_img)
            filtered_ids.append(uid)
    
    excluded_users = set(user_image_count.keys()) - set(eligible_users.keys())
    if excluded_users:
        print(f"\n⚠️  Excluded users (insufficient samples):")
        for uid in sorted(excluded_users):
            cnt = user_image_count[uid]
            print(f"      User {uid}: {cnt} images (need {MIN_IMAGES_PER_USER})")
    
    print(f"\n✅ TRAINING DATA SUMMARY:")
    print(f"   Users eligible for training: {len(eligible_users)}")
    print(f"   Training samples (with augmentation): {len(filtered_faces)}")
    print(f"   Average samples per user: {len(filtered_faces)//len(eligible_users)}")
    
    if len(skipped_images) > 0:
        print(f"\n⚠️  Skipped {len(skipped_images)} images:")
        for filename, reason in skipped_images[:10]:  # Show first 10
            print(f"      {filename}: {reason}")
        if len(skipped_images) > 10:
            print(f"      ... and {len(skipped_images)-10} more")
    
    # Train the model
    print(f"\n🤖 TRAINING MODEL:")
    print(f"   Training on {len(filtered_faces)} samples from {len(eligible_users)} users...")
    
    try:
        recognizer.train(filtered_faces, np.array(filtered_ids, dtype=np.int32))
        print(f"   ✓ Model trained successfully")
    except Exception as e:
        print(f"\n❌ TRAINING ERROR: {e}")
        return False
    
    # Save model
    os.makedirs("TrainingImageLabel", exist_ok=True)
    model_path = "TrainingImageLabel/Trainner.yml"
    
    try:
        recognizer.save(model_path)
        model_size = os.path.getsize(model_path)
        print(f"   ✓ Model saved: {model_path} ({model_size} bytes)")
    except Exception as e:
        print(f"\n❌ ERROR saving model: {e}")
        return False
    
    # Summary
    print(f"\n" + "="*70)
    print(f"✅ TRAINING COMPLETE!")
    print(f"="*70)
    print(f"\n📊 Final Statistics:")
    print(f"   Trained Users: {len(eligible_users)}")
    for uid in sorted(eligible_users.keys()):
        print(f"      • User {uid}: {eligible_users[uid]} images")
    print(f"\n📁 Model Location: {model_path}")
    print(f"🎯 Recognition Threshold: {RECOGNIZER_THRESHOLD}")
    print(f"   (Lower values = stricter matching)")
    print(f"\n💡 Next steps:")
    print(f"   1. Go to Enroll page and test face recognition")
    print(f"   2. Adjust threshold if needed (lower = stricter, higher = lenient)")
    print(f"   3. Capture more images if recognition is unreliable")
    
    return True

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
