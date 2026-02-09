import sqlite3

db = sqlite3.connect("database/attendance.db")

db.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    description TEXT,
    status TEXT DEFAULT 'Open',
    date TEXT
)
""")

db.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rating INTEGER,
    message TEXT,
    date TEXT
)
""")

db.commit()
db.close()

print("Admin tables created")
