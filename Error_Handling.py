import sqlite3

conn = sqlite3.connect("stock_data.db")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

conn.close()

print("Available tables:", tables)