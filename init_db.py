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

# ------------------------------
# Payroll Settings Table
# ------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS payroll_settings (
    user_id INTEGER PRIMARY KEY,
    monthly_salary REAL DEFAULT 0,
    hourly_rate REAL DEFAULT 0,
    workdays_per_month INTEGER DEFAULT 0,
    standard_hours REAL DEFAULT 8,
    overtime_multiplier REAL DEFAULT 1.5,
    deduction_per_absent_day REAL DEFAULT 0,
    bonus_amount REAL DEFAULT 0,
    updated_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
""")

# ------------------------------
# Payroll History Table
# ------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS payroll_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    month TEXT NOT NULL,
    year INTEGER NOT NULL,
    present_days INTEGER DEFAULT 0,
    absent_days INTEGER DEFAULT 0,
    expected_workdays INTEGER DEFAULT 0,
    total_hours REAL DEFAULT 0,
    overtime_hours REAL DEFAULT 0,
    base_salary REAL DEFAULT 0,
    overtime_pay REAL DEFAULT 0,
    deductions REAL DEFAULT 0,
    bonuses REAL DEFAULT 0,
    gross_salary REAL DEFAULT 0,
    net_salary REAL DEFAULT 0,
    generated_at TEXT,
    status TEXT DEFAULT 'generated',
    UNIQUE(user_id, month),
    FOREIGN KEY (user_id) REFERENCES users(id)
)
""")

# ------------------------------
# Leave Requests Table
# ------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS leave_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    leave_type TEXT DEFAULT 'General',
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    total_days INTEGER NOT NULL,
    reason TEXT NOT NULL,
    request_status TEXT DEFAULT 'Pending',
    admin_comment TEXT,
    applied_at TEXT,
    reviewed_at TEXT,
    reviewed_by TEXT
)
""")

db.commit()
db.close()

print("Database initialized successfully")
