"""
Improved face detection and preprocessing utilities
Handles reliable face detection, preprocessing, and image normalization
"""
import cv2
import numpy as np
import os
from collections import defaultdict

# Optimized cascade classifier paths
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
FACE_CASCADE_ALT_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")

# Face detection and training constants
FACE_WIDTH = 200
FACE_HEIGHT = 200
MIN_FACE_SIZE = (60, 60)
MAX_FACE_SIZE = (500, 500)

# Detection parameters
DETECT_SCALE_FACTOR = 1.2
DETECT_MIN_NEIGHBORS = 5
SMALL_FRAME_SCALE = 0.6  # Detect on downsampled frame for speed

# Recognition parameters
CONFIDENCE_THRESHOLD = 60  # Tuned for LBPH recognizer
MIN_STABLE_FRAMES = 3
MAX_STABLE_FRAMES = 5

TRAINING_GALLERY_CACHE = {}
TRAINING_GALLERY_CACHE_LOADED_AT = 0.0


def load_cascade_classifiers():
    """Load cascade classifiers with fallback support."""
    face_cascade = None
    eye_cascade = None
    
    # Try local path first, then fallback to OpenCV data
    for face_path in [FACE_CASCADE_PATH, FACE_CASCADE_ALT_PATH]:
        if os.path.exists(face_path):
            try:
                face_cascade = cv2.CascadeClassifier(face_path)
                if not face_cascade.empty():
                    print(f"✅ Face cascade loaded from: {face_path}")
                    break
            except Exception as e:
                print(f"⚠️ Failed to load cascade from {face_path}: {e}")
                continue
    
    if face_cascade is None or face_cascade.empty():
        print("❌ ERROR: Could not load face cascade classifier!")
        return None, None
    
    try:
        eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        if eye_cascade.empty():
            print("⚠️ Eye cascade not available")
            eye_cascade = None
    except Exception as e:
        print(f"⚠️ Could not load eye cascade: {e}")
        eye_cascade = None
    
    return face_cascade, eye_cascade


def preprocess_face(face_roi):
    """
    Preprocess face region for better recognition:
    - Apply histogram equalization
    - Normalize lighting
    - Ensure uint8 type
    """
    if face_roi is None or face_roi.size == 0:
        return None
    
    try:
        # Ensure grayscale and uint8
        if len(face_roi.shape) == 3:
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_roi
        
        face_gray = face_gray.astype(np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better results
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_preprocessed = clahe.apply(face_gray)
        
        return face_preprocessed
    except Exception as e:
        print(f"⚠️ Face preprocessing error: {e}")
        return face_roi


def resize_face_with_padding(face_roi, target_width=FACE_WIDTH, target_height=FACE_HEIGHT):
    """
    Resize face with aspect ratio preservation using padding.
    Prevents distortion that reduces recognition accuracy.
    """
    if face_roi is None or face_roi.size == 0:
        return None
    
    h, w = face_roi.shape[:2]
    
    # Calculate scale to fit within target dimensions while preserving aspect ratio
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Create canvas and center the resized image
    canvas = np.zeros((target_height, target_width), dtype=face_roi.dtype)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def prepare_face_for_model(face_roi):
    """Normalize a face crop into the 200x200 grayscale format used by the model."""
    if face_roi is None or face_roi.size == 0:
        return None

    try:
        processed = preprocess_face(face_roi)
        if processed is None or processed.size == 0:
            return None

        return resize_face_with_padding(
            processed,
            target_width=FACE_WIDTH,
            target_height=FACE_HEIGHT,
        )
    except Exception:
        return None


def _normalize_face_vector(face_img):
    if face_img is None:
        return None

    arr = np.asarray(face_img, dtype=np.float32)
    if arr.size == 0:
        return None

    arr = arr.reshape(-1)
    arr = arr - float(np.mean(arr))
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return None
    return arr / norm


def face_similarity(face_a, face_b):
    """Return cosine similarity between two aligned face images in [-1, 1]."""
    vec_a = _normalize_face_vector(face_a)
    vec_b = _normalize_face_vector(face_b)
    if vec_a is None or vec_b is None or vec_a.shape != vec_b.shape:
        return None

    return float(np.dot(vec_a, vec_b))


def load_training_face_gallery(force_refresh=False):
    """Load and cache normalized training faces grouped by user id."""
    global TRAINING_GALLERY_CACHE
    global TRAINING_GALLERY_CACHE_LOADED_AT

    now = cv2.getTickCount() / cv2.getTickFrequency()
    if not force_refresh and TRAINING_GALLERY_CACHE and (now - TRAINING_GALLERY_CACHE_LOADED_AT) < 10.0:
        return TRAINING_GALLERY_CACHE

    gallery = defaultdict(list)
    if os.path.exists("TrainingImage"):
        for filename in os.listdir("TrainingImage"):
            if not (filename.startswith("User.") and filename.endswith(".jpg")):
                continue

            parts = filename.split(".")
            if len(parts) < 4:
                continue

            try:
                user_id = int(parts[1])
            except ValueError:
                continue

            image_path = os.path.join("TrainingImage", filename)
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                if image.shape != (FACE_HEIGHT, FACE_WIDTH):
                    image = cv2.resize(image, (FACE_WIDTH, FACE_HEIGHT), interpolation=cv2.INTER_CUBIC)

                gallery[user_id].append(image)
            except Exception:
                continue

    TRAINING_GALLERY_CACHE = dict(gallery)
    TRAINING_GALLERY_CACHE_LOADED_AT = now
    return TRAINING_GALLERY_CACHE


def verify_prediction_with_gallery(face_roi, candidate_user_id, min_supporting_samples=4):
    """Reject a prediction unless it also matches the user's stored gallery.

    Returns (is_valid, debug_payload).
    """
    prepared = prepare_face_for_model(face_roi)
    if prepared is None:
        return False, {"error": "unusable_face"}

    gallery = load_training_face_gallery()
    if candidate_user_id not in gallery or not gallery[candidate_user_id]:
        return False, {"error": "candidate_gallery_missing", "candidate_user_id": candidate_user_id}

    per_user_scores = []
    candidate_scores = []

    for user_id, samples in gallery.items():
        sample_scores = []
        for sample in samples:
            score = face_similarity(prepared, sample)
            if score is not None:
                sample_scores.append(score)

        if not sample_scores:
            continue

        sample_scores.sort(reverse=True)
        best_score = float(sample_scores[0])
        avg_best_scores = float(np.mean(sample_scores[: min(3, len(sample_scores))]))
        per_user_scores.append({
            "user_id": user_id,
            "best_score": best_score,
            "avg_top_scores": avg_best_scores,
            "supporting_samples": sum(1 for score in sample_scores if score >= 0.38),
        })

        if user_id == candidate_user_id:
            candidate_scores = sample_scores

    if not per_user_scores or not candidate_scores:
        return False, {"error": "gallery_verification_unavailable", "candidate_user_id": candidate_user_id}

    per_user_scores.sort(key=lambda row: (row["best_score"], row["avg_top_scores"]), reverse=True)
    top_user = per_user_scores[0]
    runner_up = per_user_scores[1] if len(per_user_scores) > 1 else None

    candidate_best = next((row for row in per_user_scores if row["user_id"] == candidate_user_id), None)
    if candidate_best is None:
        return False, {"error": "candidate_not_scored", "candidate_user_id": candidate_user_id}

    candidate_support = sum(1 for score in candidate_scores if score >= 0.38)
    candidate_avg_top = float(np.mean(sorted(candidate_scores, reverse=True)[: min(3, len(candidate_scores))]))
    candidate_margin = candidate_best["best_score"] - (runner_up["best_score"] if runner_up else -1.0)

    is_top_match = top_user["user_id"] == candidate_user_id
    is_strong_enough = (
        candidate_best["best_score"] >= 0.50
        and candidate_avg_top >= 0.43
        and candidate_support >= min_supporting_samples
        and candidate_margin >= 0.07
        and is_top_match
    )

    return is_strong_enough, {
        "candidate_user_id": candidate_user_id,
        "candidate_best_score": round(candidate_best["best_score"], 4),
        "candidate_avg_top": round(candidate_avg_top, 4),
        "candidate_supporting_samples": candidate_support,
        "candidate_margin": round(candidate_margin, 4),
        "top_user_id": top_user["user_id"],
        "top_user_score": round(top_user["best_score"], 4),
        "runner_up_user_id": runner_up["user_id"] if runner_up else None,
        "runner_up_score": round(runner_up["best_score"], 4) if runner_up else None,
        "all_user_scores": [
            {
                "user_id": row["user_id"],
                "best_score": round(row["best_score"], 4),
                "avg_top_scores": round(row["avg_top_scores"], 4),
                "supporting_samples": row["supporting_samples"],
            }
            for row in per_user_scores[:5]
        ],
        "accepted": is_strong_enough,
    }


def predict_face_with_flip(recognizer, face_roi):
    """Predict face identity using both original and flipped crops.

    Returns a tuple of (predicted_id, confidence, debug_payload).
    If the two orientations disagree, the prediction is rejected as ambiguous
    instead of forcing a label.
    """
    prepared = prepare_face_for_model(face_roi)
    if prepared is None:
        return None, None, {"error": "unusable_face"}

    try:
        original_id, original_conf = recognizer.predict(prepared)
        flipped_id, flipped_conf = recognizer.predict(cv2.flip(prepared, 1))
    except Exception as e:
        return None, None, {"error": str(e)}

    confidence_gap = abs(float(original_conf) - float(flipped_conf))

    if original_id == flipped_id:
        chosen_id = original_id
        chosen_conf = min(original_conf, flipped_conf)
        chosen_source = "original" if original_conf <= flipped_conf else "flipped"
        flip_resolved_by_confidence = False
        ambiguous_flip = False
    else:
        return None, None, {
            "error": "ambiguous_flip_prediction",
            "original_id": original_id,
            "original_conf": float(original_conf),
            "flipped_id": flipped_id,
            "flipped_conf": float(flipped_conf),
            "confidence_gap": round(confidence_gap, 2),
            "agreement": False,
            "flip_resolved_by_confidence": False,
            "ambiguous_flip": True,
        }

    if chosen_conf > CONFIDENCE_THRESHOLD:
        return None, None, {
            "error": "low_confidence_flip_prediction",
            "original_id": original_id,
            "original_conf": float(original_conf),
            "flipped_id": flipped_id,
            "flipped_conf": float(flipped_conf),
            "chosen_source": chosen_source,
            "confidence_gap": round(confidence_gap, 2),
        }

    return chosen_id, chosen_conf, {
        "original_id": original_id,
        "original_conf": float(original_conf),
        "flipped_id": flipped_id,
        "flipped_conf": float(flipped_conf),
        "chosen_source": chosen_source,
        "confidence_gap": round(confidence_gap, 2),
        "agreement": original_id == flipped_id,
        "flip_resolved_by_confidence": flip_resolved_by_confidence,
        "ambiguous_flip": ambiguous_flip,
    }


def detect_faces_optimized(frame, cascade_classifier):
    """
    Detect faces using multi-scale detection on downsampled frame for speed.
    Returns list of (x, y, w, h) tuples sorted by size (largest first).
    """
    if cascade_classifier is None or cascade_classifier.empty():
        return []
    
    try:
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect on downsampled frame for speed
        small_gray = cv2.resize(gray, (0, 0), fx=SMALL_FRAME_SCALE, fy=SMALL_FRAME_SCALE)
        
        faces_small = cascade_classifier.detectMultiScale(
            small_gray,
            scaleFactor=DETECT_SCALE_FACTOR,
            minNeighbors=DETECT_MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
            maxSize=MAX_FACE_SIZE
        )
        
        # Scale back to original frame coordinates
        faces = []
        for (sx, sy, sw, sh) in faces_small:
            x = int(sx / SMALL_FRAME_SCALE)
            y = int(sy / SMALL_FRAME_SCALE)
            w = int(sw / SMALL_FRAME_SCALE)
            h = int(sh / SMALL_FRAME_SCALE)
            faces.append((x, y, w, h))
        
        # Sort by size (area), largest first
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        
        return faces
    except Exception as e:
        print(f"⚠️ Face detection error: {e}")
        return []


def get_best_face(frame, cascade_classifier, max_faces=1):
    """
    Get the best (usually largest) detected face.
    Returns: (x, y, w, h) or None if no faces found.
    """
    faces = detect_faces_optimized(frame, cascade_classifier)
    
    if faces:
        return faces[0]  # Returns largest face
    return None


def extract_and_prepare_face(frame, face_coords, cascade_classifier=None):
    """
    Extract face region from frame and prepare for recognition.
    Returns: preprocessed face image or None if extraction fails.
    """
    if frame is None or face_coords is None:
        return None
    
    x, y, w, h = face_coords
    
    try:
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return None
        
        # Preprocess
        face_preprocessed = preprocess_face(face_roi)
        
        if face_preprocessed is None:
            return None
        
        # Resize with padding
        face_resized = resize_face_with_padding(
            face_preprocessed,
            target_width=FACE_WIDTH,
            target_height=FACE_HEIGHT
        )
        
        return face_resized
    except Exception as e:
        print(f"⚠️ Face extraction error: {e}")
        return None


def draw_face_rectangle(frame, face_coords, label="", confidence=0, color=(0, 255, 0)):
    """
    Draw rectangle and label on face.
    Returns: modified frame.
    """
    if frame is None or face_coords is None:
        return frame
    
    x, y, w, h = face_coords
    
    try:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw label if provided
        if label:
            if confidence:
                display_label = f"{label} ({confidence:.1f}%)"
            else:
                display_label = label
            
            # Draw label background and text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_w, text_h), baseline = cv2.getTextSize(
                display_label, font, font_scale, thickness
            )
            
            label_x = max(x, 10)
            label_y = max(y - 10, 30)
            label_x2 = min(label_x + text_w + 10, frame.shape[1] - 10)
            label_y2 = min(label_y + text_h + 10, frame.shape[0])
            
            cv2.rectangle(
                frame,
                (label_x, label_y - text_h - 5),
                (label_x2, label_y2 - text_h + 15),
                (0, 0, 0),
                -1
            )
            
            cv2.putText(
                frame,
                display_label,
                (label_x + 5, label_y + 5),
                font,
                font_scale,
                color,
                thickness
            )
        
        return frame
    except Exception as e:
        print(f"⚠️ Drawing error: {e}")
        return frame


def is_profile_pose(face_roi):
    """
    Detect if face is in profile or side pose (unreliable for recognition).
    Returns: True if profile detected, False otherwise.
    """
    try:
        # Simple heuristic: check vertical edge density
        if face_roi is None or face_roi.size < 100:
            return False
        
        left_third = face_roi[:, :face_roi.shape[1]//3]
        right_third = face_roi[:, 2*face_roi.shape[1]//3:]
        
        left_edges = cv2.Canny(left_third, 50, 150)
        right_edges = cv2.Canny(right_third, 50, 150)
        
        left_density = np.sum(left_edges) / float(left_edges.size) if left_edges.size > 0 else 0
        right_density = np.sum(right_edges) / float(right_edges.size) if right_edges.size > 0 else 0
        
        # Sharp difference suggests profile
        return abs(left_density - right_density) > 0.015
    except:
        return False


def is_blurry(face_roi, threshold=100):
    """
    Detect if face image is blurry using Laplacian variance.
    Returns: True if blurry, False otherwise.
    """
    try:
        if face_roi is None or face_roi.size < 100:
            return True
        
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance < threshold
    except:
        return True
