import os
import sqlite3
import shutil

print("🧹 Clearing old data...")

# Clear training images
if os.path.exists("TrainingImage"):
    shutil.rmtree("TrainingImage")
    print("✅ Cleared TrainingImage folder")

# Clear training model
if os.path.exists("TrainingImageLabel"):
    shutil.rmtree("TrainingImageLabel")
    print("✅ Cleared TrainingImageLabel folder")

# Clear attendance records and reset to valid emp_id values only
conn = sqlite3.connect("database/attendance.db")
cur = conn.cursor()

# Delete records with invalid emp_ids
invalid_count = cur.execute("SELECT COUNT(*) FROM attendance WHERE emp_id NOT IN (SELECT id FROM users)").fetchone()[0]
cur.execute("DELETE FROM attendance WHERE emp_id NOT IN (SELECT id FROM users)")
conn.commit()
print(f"✅ Deleted {invalid_count} invalid attendance records")

# Check remaining records
remaining = cur.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
print(f"Remaining valid attendance records: {remaining}")

conn.close()
os.makedirs("TrainingImage", exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)
print("\n✅ System cleaned and ready for fresh enrollment")
