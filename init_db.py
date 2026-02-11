# init_db.py

import sqlite3

db = sqlite3.connect("database/attendance.db")
cursor = db.cursor()

# ------------------------------
# Complaints Table (existing)
# ------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    description TEXT,
    status TEXT DEFAULT 'Open',
    date TEXT
)
""")

# ------------------------------
# Feedback Table (existing)
# ------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rating INTEGER,
    message TEXT,
    date TEXT
)
""")

# ------------------------------
# Employees Table (NEW)
# ------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    emp_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT,
    hourly_rate REAL DEFAULT 0
)
""")

# ------------------------------
# Attendance Table (UPGRADED)
# ------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id TEXT NOT NULL,
    date TEXT NOT NULL,
    day TEXT,
    checkin_time TEXT,
    checkout_time TEXT,
    worked_hours REAL,
    status TEXT DEFAULT 'Present',
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
)
""")

db.commit()
db.close()

print("Database initialized successfully")
