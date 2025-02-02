##############################################
# script/database_pipe.py
##############################################
import csv
import sqlite3
import os
import logging

# ===== 日誌設定 =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 設定資料庫和 CSV 檔案路徑
database_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'database.db')  # DB 路徑
csv_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'House_Rent_Info.csv')  # CSV 路徑

def get_db_connection():
    """建立並回傳資料庫連接"""
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    return conn

def define_sql():
    """定義並回傳建立 'listings' 表的 SQL 語句"""
    # 使用 'ID' 作為 PRIMARY KEY，避免使用 AUTOINCREMENT，並將 'Name' 改為 'title'
    sql = """
    CREATE TABLE IF NOT EXISTS listings (
        ID INTEGER PRIMARY KEY,
        title TEXT,
        Price REAL,
        Size REAL,
        Age REAL,
        Floors INTEGER,
        Bedroom INTEGER,
        City TEXT,
        Location TEXT,
        HouseType TEXT,
        Pattern TEXT,
        Tags TEXT,
        Environment TEXT,
        Url TEXT,
        Image_name TEXT,
        CP_value REAL
    );
    """
    return sql

def create_table():
    """建立 'listings' 表（如果尚未存在）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(define_sql())
    conn.commit()
    conn.close()
    logging.info("Table 'listings' has been created/updated if not exists.")

def insert_or_replace_listing(row):
    """根據 CSV 資料插入或替換資料庫中的房源資料"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 取得並處理 CSV 中的欄位
    try:
        ID = int(row.get('ID', 0))
        title = row.get('Name', '').strip()  # 將 'Name' 改為 'title'
        Price = float(row.get('Price', 0))
        Size = float(row.get('Size', 0))
        Age = float(row.get('Age', 0))
        Floors = int(float(row.get('Floors', 0)))
        Bedroom = int(float(row.get('Bedroom', 0)))
        City = row.get('City', '').strip()
        Location = row.get('Location', '').strip()
        HouseType = row.get('HouseType', '').strip()
        Pattern = row.get('Pattern', '').strip()
        Tags = row.get('Tags', '').strip()
        Environment = row.get('Environment', '').strip()
        Url = row.get('Url', '').strip()
        Image_name = row.get('Image_name', '').strip()
        CP_value = float(row.get('CP_value', 0))

        # 使用 INSERT OR REPLACE 來覆蓋已有的 ID 資料，並將 'Name' 改為 'title'
        cursor.execute("""
            INSERT OR REPLACE INTO listings (
                ID, title, Price, Size, Age,
                Floors, Bedroom, City, Location, HouseType,
                Pattern, Tags, Environment, Url,
                Image_name, CP_value
            ) VALUES (?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?,
                      ?, ?, ?, ?,
                      ?, ?)
        """, (
            ID, title, Price, Size, Age,
            Floors, Bedroom, City, Location, HouseType,
            Pattern, Tags, Environment, Url,
            Image_name, CP_value
        ))
        conn.commit()
        logging.info(f"Inserted/Replaced listing ID={ID}: {title}")
    except Exception as e:
        logging.error(f"Error inserting listing ID={row.get('ID', 'Unknown')}: {e}")
    finally:
        conn.close()

def import_csv_to_db():
    """將 CSV 資料導入資料庫，並根據 ID 覆蓋現有資料"""
    create_table()
    if not os.path.exists(csv_file_path):
        logging.error(f"CSV file not found: {csv_file_path}")
        return

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                insert_or_replace_listing(row)
                print(f"Processed: ID={row.get('ID', 'Unknown')}, Title={row.get('Name', 'NoTitle')}")
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")

def rename_column_if_exists():
    """
    如果 'listings' 表格中存在 'Name' 欄位，則將其重命名為 'title'。
    SQLite 不直接支持重命名欄位，因此需要創建一個新表，複製資料，然後替換原表。
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 檢查 'Name' 欄位是否存在
    cursor.execute("PRAGMA table_info(listings);")
    columns = [row['name'] for row in cursor.fetchall()]
    if 'Name' in columns:
        logging.info("Renaming 'Name' column to 'title' in 'listings' table.")

        # 創建一個臨時表，使用正確的欄位名稱
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS listings_temp (
                ID INTEGER PRIMARY KEY,
                title TEXT,
                Price REAL,
                Size REAL,
                Age REAL,
                Floors INTEGER,
                Bedroom INTEGER,
                City TEXT,
                Location TEXT,
                HouseType TEXT,
                Pattern TEXT,
                Tags TEXT,
                Environment TEXT,
                Url TEXT,
                Image_name TEXT,
                CP_value REAL
            );
        """)

        # 複製資料到臨時表，將 'Name' 資料插入 'title' 欄位
        cursor.execute("""
            INSERT OR REPLACE INTO listings_temp (
                ID, title, Price, Size, Age,
                Floors, Bedroom, City, Location, HouseType,
                Pattern, Tags, Environment, Url,
                Image_name, CP_value
            )
            SELECT 
                ID, Name as title, Price, Size, Age,
                Floors, Bedroom, City, Location, HouseType,
                Pattern, Tags, Environment, Url,
                Image_name, CP_value
            FROM listings;
        """)

        # 刪除原始表格
        cursor.execute("DROP TABLE listings;")

        # 將臨時表格重新命名為原始表格名稱
        cursor.execute("ALTER TABLE listings_temp RENAME TO listings;")

        conn.commit()
        logging.info("'Name' 欄位已成功重命名為 'title'.")
    else:
        logging.info("No 'Name' column found in 'listings' table. No renaming needed.")

    conn.close()

if __name__ == "__main__":
    rename_column_if_exists()  # 先嘗試重命名欄位
    import_csv_to_db()
