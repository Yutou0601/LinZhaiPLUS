import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE_PATH = "data/database.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row   
    return conn


def create_users_table():
    """建立 users 表格"""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            email TEXT,
            role TEXT DEFAULT 'user',
            browse_list INT
        )
    ''')
    conn.commit()
    conn.close()

def drop_table(table_name):
    conn = get_db_connection()
    try:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        print(f"表格 '{table_name}' 已刪除或不存在。")
    except sqlite3.Error as e:
        print(f"刪除表格 '{table_name}' 時出現錯誤: {e}")
    finally:
        conn.close()


def register_user(username, password, email):
    """新增用戶"""
    import random
    create_users_table()
    conn = get_db_connection()
    hashed_password = generate_password_hash(password)
    try:
        conn.execute(
            "INSERT INTO users (username, password, email , browse_list) VALUES (?, ?, ?, ?)",
            (username, hashed_password, email , random.randint(0,100))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    """驗證用戶登入"""
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()
    if user and check_password_hash(user['password'], password):
        return user
    return None
 

if __name__ == '__main__':
    drop_table(table_name="users")