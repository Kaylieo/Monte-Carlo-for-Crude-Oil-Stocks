import sqlite3
import pandas as pd

conn = sqlite3.connect("stock_data.db")
query = "SELECT * FROM BP LIMIT 5"
df = pd.read_sql(query, conn)
conn.close()

print(df.head())  # Should now have a 'Date' column
