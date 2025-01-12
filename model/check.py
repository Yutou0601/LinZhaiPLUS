import sqlite3

database_path = "data/database.db"  # 根據實際路徑調整

conn = sqlite3.connect(database_path)
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(listings);")
columns = cursor.fetchall()

print("欄位名稱：")
for column in columns:
    print(column[1])  # 欄位名稱在第二個元素

conn.close()
