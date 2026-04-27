import sqlite3

db = sqlite3.connect("database/attendance.db")

db.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    department TEXT,
    role TEXT DEFAULT 'employee',
    created_at TEXT
)
""")

db.commit()
db.close()

print("✅ Users table (full profile) ready")
