from flask import Flask, render_template, request, redirect, url_for, session, flash
from model.knn import knn_similar_listings
import sqlite3
from model.map_func import generate_map
from model.db_handler import *
from model.CP_estimate import predict_cp_value, load_label_encoders
from model.xgb import predict_price
import pandas as pd
import os
from model.CP_estimate import predict_cp_value, load_label_encoders
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
        # 代表預測失敗，您可以選擇回傳 0 或顯示警告
        logging.warning("predict_cp_value 回傳 None，無法取得 CP Value")
        return 0
    
    return cp_value[0]  # 取出預測值

# 查詢房屋清單 
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

# 獲取可選的城市和屋型
def get_filter_options():
    conn = get_db_connection()
    cities = conn.execute("SELECT DISTINCT city FROM listings").fetchall()
    types = conn.execute("SELECT DISTINCT house_type FROM listings").fetchall()
    conn.close()
    
    # 把查詢結果轉換成簡單的 list 形式
    cities = [city['city'] for city in cities]
    types = [type_['house_type'] for type_ in types]
    
    return cities, types

# 瀏覽頁面路由
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
    
    # 查詢資料庫資料
    listings = query_listings(filter_type, filter_city, min_rent, max_rent)
    
    # 取得篩選條件選項
    cities, types = get_filter_options()

    return render_template('browse.html', listings=listings, filter_type=filter_type, filter_city=filter_city, 
                           min_rent=min_rent, max_rent=max_rent, cities=cities, types=types)

# 根路徑路由
@app.route('/')
def home():
    return render_template('home.html')  # 或使用 redirect 到 '/browse'

# 詳情頁面路由
@app.route('/detail/<int:id>')
def detail(id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM listings WHERE id = ?", (id,)).fetchone()
    if not row:
        conn.close()
        return render_template('house_not_exist.html')

    listing = dict(row)

    # --------------------------------------------------------
    # (A) 先確保 city / house_type 字串化，再把 '0' 或空字串改為 'Unknown'
    # --------------------------------------------------------
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

    # 計算區域平均價格
    avg_district_price = get_avg_district_price(district, conn)
    
    # 預測 CP_value
    predicted_cp_value = calculate_cp_value(listing)
    
    # 預測 價格
    predicted_price = predict_price(listing)

    # 取得所有同城市的房屋列表
    all_listings = conn.execute(
        "SELECT * FROM listings WHERE city = ? AND id != ?",
        (city, id)
    ).fetchall()
    all_listings = [dict(l) for l in all_listings]

    same_district_listings = [house for house in all_listings if house['district'] == district]
    recommended_listings = knn_similar_listings(all_listings, listing)

    conn.close()

    address = listing['location'].split('/')[0]
    map_html, place_count = generate_map(address)

    return render_template(
        'detail.html',
        listing=listing,
        similar_listings=all_listings,
        predicted_cp_value=predicted_cp_value,
        predicted_price=predicted_price,
        avg_district_price=round(avg_district_price, 3),
        map_html=map_html,
        place_count=place_count,
        district_listings=same_district_listings,   
        recommended_listings=recommended_listings,  
    )
    

def get_avg_district_price(district, conn):
    """計算該區域的平均價格"""
    avg_price = conn.execute("SELECT AVG(price) FROM listings WHERE district = ?", (district,)).fetchone()[0]
    return avg_price

# Search Implement
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

# Recommender Route
@app.route('/recommender', methods=['GET', 'POST'])
def recommender():
    try:
        # 假設 'house_index' 在 session 中是當前房源的 ID
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

        # 取得所有同城市的房屋列表
        all_listings = conn.execute(
            "SELECT * FROM listings WHERE city = ? AND id != ?",
            (city, id)
        ).fetchall()
        all_listings = [dict(l) for l in all_listings]

        # KNN推薦：同城市的房源 
        recommended_listings = knn_similar_listings(all_listings, listing)
        conn.close()

        # 從資料庫中讀取 CP_value 並排序
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

        # 將 DataFrame 轉換為字典列表
        sorted_listings = sorted_df.to_dict(orient='records')

        # ================================
        # (1) 在此處「去重」：
        #     以 (Name, Price, Image_name) 為判斷依據
        # ================================
        unique_dict = {}
        for item in sorted_listings:
            # 將標題、價格、圖片三者組合成 key
            the_key = (
                item.get('Name'),
                item.get('Price'),
                item.get('Image_name')
            )
            if the_key not in unique_dict:
                unique_dict[the_key] = item

        # 只保留第一個遇到的
        unique_list = list(unique_dict.values())

    except Exception as e:
        logging.error(f"推薦頁面發生錯誤：{e}")
        flash(f"發生錯誤：{str(e)}", 'danger')
        unique_list = []

    # 將「去重後」的結果傳給模板
    return render_template('recommender.html', listings=unique_list)



# Register, Login, Logout, Manage Routes
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
            session['house_index'] = user['browse_list']  # 確保 'browse_list' 是正確的欄位
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
    
    user_id  = session['user_id']
    username = session['username']
    return render_template('manage.html', username=username)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
