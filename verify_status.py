import os
import sqlite3

print("=== SYSTEM STATUS ===\n")

# Check training images
training_count = len([f for f in os.listdir("TrainingImage") if f.startswith("User.")])
print(f"Training images: {training_count}")

# Check model
model_exists = os.path.exists("TrainingImageLabel/Trainner.yml")
print(f"Model exists: {'✅ YES' if model_exists else '❌ NO'}")

# Check database users
conn = sqlite3.connect("database/attendance.db")
print("\nDatabase Users:")
for row in conn.execute("SELECT id, name FROM users"):
    print(f"  ID: {row[0]}, Name: {row[1]}")
conn.close()

print("\n✅ Ready for enrollment" if training_count == 0 and not model_exists else "\n⚠️ System not clean")
