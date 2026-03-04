import os
from collections import defaultdict

print("=== TRAINING DATA VERIFICATION ===\n")

if not os.path.exists("TrainingImage"):
    print("❌ TrainingImage folder missing!")
    exit()

files = os.listdir("TrainingImage")
user_images = defaultdict(list)

for f in files:
    if f.startswith("User."):
        parts = f.split(".")
        user_id = parts[1]
        user_images[user_id].append(f)

print("Training Images per User:")
for uid in sorted(user_images.keys()):
    count = len(user_images[uid])
    status = "✅" if count >= 15 else "⚠️"
    print(f"  User {uid}: {count}/20 images {status}")

model_exists = os.path.exists("TrainingImageLabel/Trainner.yml")
print(f"\nModel Trained: {'✅ YES' if model_exists else '❌ NO'}")

if len(user_images) >= 2 and all(len(v) >= 15 for v in user_images.values()):
    print("\n✅ READY TO TEST FACE RECOGNITION")
else:
    print("\n⚠️ More enrollment needed - ensure at least 2 users with 15+ images each")
