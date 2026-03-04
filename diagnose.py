import sqlite3
import os
from collections import defaultdict

db_path = "database/attendance.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Check users
print("=== DATABASE USERS ===")
users = cur.execute("SELECT id, name, role FROM users").fetchall()
for user in users:
    print(f"ID: {user[0]}, Name: {user[1]}, Role: {user[2]}")

# Check training images
print("\n=== TRAINING IMAGES ===")
if os.path.exists("TrainingImage"):
    files = os.listdir("TrainingImage")
    print(f"Total training images: {len(files)}")
    # Group by user
    user_images = defaultdict(list)
    for f in files:
        if f.startswith("User."):
            parts = f.split(".")
            user_id = parts[1]
            user_images[user_id].append(f)
    
    for uid in sorted(user_images.keys()):
        print(f"User {uid}: {len(user_images[uid])} images")
else:
    print("TrainingImage folder not found")

# Check model
print("\n=== MODEL FILE ===")
model_path = "TrainingImageLabel/Trainner.yml"
if os.path.exists(model_path):
    print(f"✅ Model exists: {model_path}")
    print(f"Size: {os.path.getsize(model_path)} bytes")
else:
    print("❌ Model does not exist")

# Check recent attendance records
print("\n=== RECENT ATTENDANCE RECORDS ===")
records = cur.execute("""
    SELECT id, user_id, date, checkin_time, checkout_time 
    FROM attendance 
    ORDER BY id DESC 
    LIMIT 10
""").fetchall()

for rec in records:
    print(f"ID: {rec[0]}, User: {rec[1]}, Date: {rec[2]}, Check-In: {rec[3]}, Check-Out: {rec[4]}")

conn.close()
