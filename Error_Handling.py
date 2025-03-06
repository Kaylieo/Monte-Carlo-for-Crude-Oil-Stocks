import sqlite3

conn = sqlite3.connect("stock_data.db")
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(XOM)")
columns = cursor.fetchall()

for col in columns:
    print(col)

conn.close()