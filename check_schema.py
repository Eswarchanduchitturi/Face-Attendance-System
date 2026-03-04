import sqlite3

conn = sqlite3.connect("database/attendance.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Get schema of attendance table
print("=== ATTENDANCE TABLE SCHEMA ===")
schema = cur.execute("PRAGMA table_info(attendance)").fetchall()
for col in schema:
    print(f"Column: {col[1]}, Type: {col[2]}")

print("\n=== ATTENDANCE RECORDS (last 5) ===")
records = cur.execute("""
    SELECT * FROM attendance ORDER BY id DESC LIMIT 5
""").fetchall()

if records:
    for i, rec in enumerate(records):
        print(f"\nRecord {i+1}:")
        for key in dict(rec).keys():
            print(f"  {key}: {dict(rec)[key]}")
else:
    print("No records found")

conn.close()
