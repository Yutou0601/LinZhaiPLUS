from flask import Flask, render_template, request, redirect, url_for, session, flash
from model.knn import knn_similar_listings
import sqlite3
import subprocess, sys
from apscheduler.schedulers.background import BackgroundScheduler
from model.map_func import generate_map
from model.db_handler import *
from model.CP_estimate import predict_cp_value, load_label_encoders
from model.xgb import predict_cp_value
import pandas as pd
import os
from model.CP_estimate import predict_cp_value, load_label_encoders
from apscheduler.schedulers.background import BackgroundScheduler
import logging

app = Flask(__name__)
app.secret_key = '0000'  

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 載入模型和 LabelEncoder
cp_model_path = os.path.join(app.root_path, 'model', 'trained_model.xgb')
le_city_path = os.path.join(app.root_path, 'model', 'le_city.pkl')
le_house_type_path = os.path.join(app.root_path, 'model', 'le_house_type.pkl')

le_city, le_house_type = load_label_encoders()

def get_db_connection():
    database_path = os.path.join(app.root_path, 'data', 'database.db')
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row   
    return conn

def calculate_cp_value(listing):
    """
    計算房源的 CP_value
    """
    cp_value = predict_cp_value(
        listing,
        model_path=cp_model_path,
        le_city_path=le_city_path,
        le_house_type_path=le_house_type_path
    )
    
    if cp_value is None:
        logging.warning("predict_cp_value 回傳 None，無法取得 CP Value")
        return 0
    return cp_value[0]  # 取出預測值

def query_listings(filter_type=None, filter_city=None, min_rent=None, max_rent=None):
    conn = get_db_connection()
    query = """
        SELECT id, title, description, image, house_type, city, price, location, CP_value, district
        FROM listings
        WHERE 1=1
    """
    params = []
    if filter_type:
        query += " AND house_type = ?"
        params.append(filter_type)
    if filter_city:
        query += " AND city = ?"
        params.append(filter_city)
    if min_rent:
        query += " AND price >= ?"
        params.append(min_rent)
    if max_rent:
        query += " AND price <= ?"
        params.append(max_rent)
    
    listings = conn.execute(query, params).fetchall()
    conn.close()
    return listings

def get_filter_options():
    conn = get_db_connection()
    cities = conn.execute("SELECT DISTINCT city FROM listings").fetchall()
    types = conn.execute("SELECT DISTINCT house_type FROM listings").fetchall()
    conn.close()
    cities = [city['city'] for city in cities]
    types = [type_['house_type'] for type_ in types]
    return cities, types

@app.route('/browse', methods=['GET', 'POST'])
def browse():
    if request.method == 'POST':
        filter_type = request.form.get('type')
        filter_city = request.form.get('city')
        min_rent = request.form.get('min_rent')
        max_rent = request.form.get('max_rent')
    else:
        filter_type = None
        filter_city = None
        min_rent = None
        max_rent = None
    
    listings = query_listings(filter_type, filter_city, min_rent, max_rent)
    cities, types = get_filter_options()
    return render_template('browse.html', listings=listings, filter_type=filter_type, filter_city=filter_city, 
                           min_rent=min_rent, max_rent=max_rent, cities=cities, types=types)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detail/<int:id>')
def detail(id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM listings WHERE id = ?", (id,)).fetchone()
    if not row:
        conn.close()
        return render_template('house_not_exist.html')

    listing = dict(row)
    city_str = str(listing.get('city', '')).strip()
    house_str = str(listing.get('house_type', '')).strip()
    if city_str in ['0', '']:
        city_str = 'Unknown'
    if house_str in ['0', '']:
        house_str = 'Unknown'

    listing['city'] = city_str
    listing['house_type'] = house_str

    city = listing['city']
    district = listing['district']
    avg_district_price = get_avg_district_price(district, conn)
    predicted_cp_value = calculate_cp_value(listing)
    predicted_price = predict_price(listing)

    # 使用 LIMIT 限制返回的 all_listings 數量
    all_listings = conn.execute(
        "SELECT * FROM listings WHERE city = ? AND id != ? LIMIT 50",  # 假設最多返回 50 條
        (city, id)
    ).fetchall()
    all_listings = [dict(l) for l in all_listings]

    # 限制 same_district_listings 為最多 210 條
    same_district_listings = [house for house in all_listings if house['district'] == district][:20]

    # 取得推薦列表並限制為最多 20 條
    recommended_listings = knn_similar_listings(all_listings, listing)[:20]

    conn.close()

    address = listing['location'].split('/')[0]
    map_html, place_count = generate_map(address)

    return render_template(
        'detail.html',
        listing=listing,
        similar_listings=all_listings[:10],  # 假設您希望顯示最多 10 個相似列表
        predicted_cp_value=predicted_cp_value,
        predicted_price=predicted_price,
        avg_district_price=round(avg_district_price, 3),
        map_html=map_html,
        place_count=place_count,
        district_listings=same_district_listings,   
        recommended_listings=recommended_listings,  
    )


def get_avg_district_price(district, conn):
    avg_price = conn.execute("SELECT AVG(price) FROM listings WHERE district = ?", (district,)).fetchone()[0]
    return avg_price

@app.route('/search', methods=['GET', 'POST'])
def search():
    search_query = request.form.get('query', '').strip() if request.method == 'POST' else ''
    listings = []
    if search_query:
        conn = get_db_connection()
        query = """
            SELECT id, title, description, image, house_type, city, price, location, CP_value, district
            FROM listings
            WHERE title LIKE ? OR description LIKE ? OR city LIKE ?
        """
        search_term = f"%{search_query}%"
        listings = conn.execute(query, (search_term, search_term, search_term)).fetchall()
        conn.close()
    return render_template('search.html', query=search_query, listings=listings)

@app.route('/recommender', methods=['GET', 'POST'])
def recommender():
    try:
        id = session.get('house_index')
        if not id:
            flash('找不到相關的房源資訊。', 'warning')
            return redirect(url_for('home'))
        
        conn = get_db_connection()
        listing = conn.execute("SELECT * FROM listings WHERE id = ?", (id,)).fetchone()
        if not listing:
            conn.close()
            return render_template('house_not_exist.html')
        listing = dict(listing)
        city = listing['city']
        district = listing['district']

        all_listings = conn.execute(
            "SELECT * FROM listings WHERE city = ? AND id != ?",
            (city, id)
        ).fetchall()
        all_listings = [dict(l) for l in all_listings]
        recommended_listings = knn_similar_listings(all_listings, listing)
        conn.close()

        conn = sqlite3.connect(os.path.join(app.root_path, 'data', 'database.db'))
        conn.row_factory = sqlite3.Row
        query = """
            SELECT 
                title        AS Name, 
                description  AS description, 
                price        AS Price, 
                size         AS Size, 
                age          AS Age, 
                floors       AS Floors, 
                bedroom      AS Bedroom, 
                city         AS City, 
                location     AS Location, 
                house_type   AS HouseType, 
                pattern      AS Pattern, 
                environment  AS Environment, 
                url          AS Url, 
                image        AS Image_name, 
                CP_value, 
                id
            FROM listings
            ORDER BY CP_value DESC
        """
        sorted_df = pd.read_sql_query(query, conn)
        conn.close()

        sorted_listings = sorted_df.to_dict(orient='records')
        unique_dict = {}
        for item in sorted_listings:
            the_key = (
                item.get('Name'),
                item.get('Price'),
                item.get('Image_name')
            )
            if the_key not in unique_dict:
                unique_dict[the_key] = item
        unique_list = list(unique_dict.values())
    except Exception as e:
        logging.error(f"推薦頁面發生錯誤：{e}")
        flash(f"發生錯誤：{str(e)}", 'danger')
        unique_list = []
    return render_template('recommender.html', listings=unique_list)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """註冊頁面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        if register_user(username, password, email):
            flash('註冊成功！請登入。', 'success')
            return redirect(url_for('login'))
        else:
            flash('帳號已被使用！請選擇其他帳號。', 'danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """登入頁面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = verify_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['house_index'] = user['browse_list']
            flash('登入成功！', 'success')
            return redirect(url_for('manage'))
        else:
            flash('帳號或密碼錯誤', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    """登出"""
    session.clear()
    flash('您已成功登出', 'success')
    return redirect(url_for('login'))

@app.route('/manage')
def manage():
    """租屋管理頁面"""
    if 'user_id' not in session:
        flash('請先登入！', 'warning')
        return redirect(url_for('login'))
    return render_template('manage.html', username=session['username'])

# ------------------------------------
# (A) 後台任務：跑爬蟲 + 匯入資料庫 + 訓練模型
# ------------------------------------
def run_crawler_pipeline_training():
    try:
        logging.info("[排程任務] 開始執行爬蟲並生成 House_Rent_Info.csv")

        # 1) 指定 Crawler_Info.py 位於 model 目錄下
        crawler_path = os.path.join(app.root_path, "model", "Crawler_Info.py")
        subprocess.run([sys.executable, crawler_path], check=True)

        logging.info("[排程任務] 爬蟲完成！開始導入資料庫 (database_pipe.py)")

        # 2) 同樣指定 database_pipe.py 在 script/ 目錄下
        dbpipe_path = os.path.join(app.root_path, "script", "database_pipe.py")
        subprocess.run([sys.executable, dbpipe_path], check=True)

        logging.info("[排程任務] 匯入資料完成！開始執行 CP_estimate.py (訓練並更新 CP_value)")

        # 3) CP_estimate.py 在 model/ 下
        cpest_path = os.path.join(app.root_path, "model", "CP_estimate.py")
        subprocess.run([sys.executable, cpest_path], check=True)

        logging.info("[排程任務] CP_value 更新完成！本次自動化流程結束。")

    except subprocess.CalledProcessError as e:
        logging.error(f"[排程任務] 執行子程序失敗：{e}")
    except Exception as ex:
        logging.error(f"[排程任務] 發生未知錯誤：{ex}")

# ------------------------------------
# (B) 排程初始化 → 每小時整點觸發
# ------------------------------------
def init_scheduler():
    """
    利用 APScheduler 於 Flask 啟動後，開啟背景排程。
    每3小時執行一次後台任務 run_crawler_pipeline_training。
    """
    logging.info("啟動 APScheduler 排程器...")
    scheduler = BackgroundScheduler()
    
    # 每3小時執行一次
    scheduler.add_job(
        run_crawler_pipeline_training,
        trigger='interval',
        hours=3,
        id='crawler_pipeline_job',  # 為任務指定一個唯一的ID
        replace_existing=True  # 如果有相同ID的任務，則替換
    )
    
    scheduler.start()
    logging.info("APScheduler 排程器已啟動，每3小時執行一次。")


@app.route('/manual_run')
def manual_run():
    """手動觸發整個後台流程（爬蟲→匯入→訓練）"""
    run_crawler_pipeline_training()
    return "手動執行完畢。"

if __name__ == '__main__':
    init_scheduler()
    app.run(debug=True, port=5001)
