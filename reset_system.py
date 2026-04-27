import os
import shutil
import sqlite3


ROOT_DB_PATH = os.path.join("database", "attendance.db")
GENERATED_PATHS = [
    "TrainingImage",
    "TrainingImageLabel",
    os.path.join("static", "attendance_images"),
]


def remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"✅ Removed folder: {path}")
    elif os.path.isfile(path):
        os.remove(path)
        print(f"✅ Removed file: {path}")


def recreate_database(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA foreign_keys = ON")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        role TEXT DEFAULT 'employee'
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        emp_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT,
        hourly_rate REAL DEFAULT 0
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        description TEXT,
        status TEXT DEFAULT 'Open',
        date TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rating INTEGER,
        message TEXT,
        date TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_id TEXT NOT NULL,
        date TEXT NOT NULL,
        day TEXT,
        checkin_time TEXT,
        checkout_time TEXT,
        worked_hours REAL,
        status TEXT DEFAULT 'Present',
        checkin_image TEXT,
        checkout_image TEXT,
        FOREIGN KEY (emp_id) REFERENCES users(id)
    )
    """)

    cur.execute("""
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

    cur.execute("""
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

    cur.execute("""
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
        reviewed_by TEXT,
        FOREIGN KEY (emp_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()


print("🧹 Clearing old data...")

for path in GENERATED_PATHS:
    remove_path(path)

if os.path.exists(ROOT_DB_PATH):
    remove_path(ROOT_DB_PATH)

recreate_database(ROOT_DB_PATH)

for path in GENERATED_PATHS:
    os.makedirs(path, exist_ok=True)
    print(f"✅ Recreated folder: {path}")

print("\n✅ System cleaned and reset to a fresh state")
print("   - training images removed")
print("   - trained model removed")
print("   - attendance photos removed")
print("   - database recreated with clean schema")
