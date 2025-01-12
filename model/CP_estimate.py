# model/CP_estimate.py

import os
import re
import json
import joblib
import sqlite3
import logging
import requests
import numpy as np
import pandas as pd
from time import sleep
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== 日誌設定 =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== 全局緩存字典 =====
address_cache = {}

# -----------------------------------------------------------------------------
# 1. 座標取得與緩存
# -----------------------------------------------------------------------------
def get_coordinates(address, geolocator, timeout=10):
    """
    使用 Nominatim 進行地理編碼，並在 address_cache 中緩存結果。
    如果地址中含有 '/'，則只取第一段進行定位。
    """
    address = fix_location(address)  # <<--- 新增這行，先修正地址

    # 其餘程式碼不變
    formatted_address = address.replace('/', ' ').strip()
    if formatted_address in address_cache:
        cached = address_cache[formatted_address]
        return cached.get('latitude'), cached.get('longitude')

    try:
        location = geolocator.geocode(formatted_address, timeout=timeout)
        if location:
            lat, lng = location.latitude, location.longitude
            address_cache[formatted_address] = {'latitude': lat, 'longitude': lng}
            return lat, lng
        else:
            logging.warning(f"無法定位地址：{address}")
            address_cache[formatted_address] = {'latitude': None, 'longitude': None}
            return None, None
    except Exception as e:
        logging.warning(f"定位地址失敗：{address}，錯誤：{e}")
        address_cache[formatted_address] = {'latitude': None, 'longitude': None}
        return None, None

    formatted_address = address.replace('/', ' ').strip()
    if formatted_address in address_cache:
        cached = address_cache[formatted_address]
        return cached.get('latitude'), cached.get('longitude')

    try:
        location = geolocator.geocode(formatted_address, timeout=timeout)
        if location:
            lat, lng = location.latitude, location.longitude
            address_cache[formatted_address] = {'latitude': lat, 'longitude': lng}
            return lat, lng
        else:
            logging.warning(f"無法定位地址：{address}")
            address_cache[formatted_address] = {'latitude': None, 'longitude': None}
            return None, None
    except Exception as e:
        logging.warning(f"定位地址失敗：{address}，錯誤：{e}")
        address_cache[formatted_address] = {'latitude': None, 'longitude': None}
        return None, None

# -----------------------------------------------------------------------------
# 2. 使用 Overpass API 搜尋附近設施
# -----------------------------------------------------------------------------
def search_osm_data(lat, lng, category, radius=1000, max_retries=3):
    TAIWAN_BOUNDS = [[20.0, 119.0], [25.5, 124.5]]
    if not (TAIWAN_BOUNDS[0][0] <= lat <= TAIWAN_BOUNDS[1][0] and
            TAIWAN_BOUNDS[0][1] <= lng <= TAIWAN_BOUNDS[1][1]):
        return []

    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node[{category}](around:{radius},{lat},{lng});
      way[{category}](around:{radius},{lat},{lng});
      relation[{category}](around:{radius},{lat},{lng});
    );
    out center;
    """
    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.get(overpass_url, params={'data': query}, timeout=10)
            if response.status_code == 200:
                elements = response.json().get("elements", [])
                return _extract_locations(elements)
            else:
                logging.error(f"OSM API 請求失敗：{response.status_code}")
        except requests.exceptions.Timeout:
            logging.error("OSM API 請求逾時。")
        except Exception as e:
            logging.error(f"OSM API 請求錯誤：{e}")
        attempt += 1
        sleep(2)
    logging.error(f"超過最大重試次數，無法取得 {category} 資料。")
    return []

def _extract_locations(elements):
    locations = []
    for e in elements:
        if 'lat' in e and 'lon' in e:
            locations.append((e['lat'], e['lon']))
        elif 'center' in e:
            c = e['center']
            if 'lat' in c and 'lon' in c:
                locations.append((c['lat'], c['lon']))
    return locations

# -----------------------------------------------------------------------------
# 3. 便利度計算
# -----------------------------------------------------------------------------
def get_convenience_features(address, geolocator):
    if address in address_cache and 'facilities' in address_cache[address]:
        return address_cache[address]['facilities']

    lat, lng = get_coordinates(address, geolocator)
    if not lat or not lng:
        facilities = _zero_facilities()
        _update_address_cache(address, facilities)
        return facilities

    PLACE_CATEGORIES = [
        "amenity","shop","leisure","tourism","education","healthcare","public_transport",
        "restaurant","fast_food","cafe","tram_stop","subway_entrance","bus_stop",
        "train_station","parking",
    ]
    convenience_dict = {}
    total_count = 0
    for category in PLACE_CATEGORIES:
        locations = search_osm_data(lat, lng, category, radius=1000)
        convenience_dict[f"{category}_count"] = len(locations)
        total_count += len(locations)

    convenience_dict["total_facilities"] = total_count
    _update_address_cache(address, convenience_dict)
    return convenience_dict

def _zero_facilities():
    return {
        "amenity_count":0,"shop_count":0,"leisure_count":0,"tourism_count":0,"education_count":0,
        "healthcare_count":0,"public_transport_count":0,"restaurant_count":0,"fast_food_count":0,
        "cafe_count":0,"tram_stop_count":0,"subway_entrance_count":0,"bus_stop_count":0,
        "train_station_count":0,"parking_count":0,"total_facilities":0
    }

def _update_address_cache(address, facilities):
    if address in address_cache:
        address_cache[address]['facilities'] = facilities
    else:
        address_cache[address] = {'facilities': facilities}

# -----------------------------------------------------------------------------
# 4. LabelEncoder 相關
# -----------------------------------------------------------------------------
def load_label_encoders():
    try:
        le_city       = joblib.load('model/le_city.pkl')
        le_house_type = joblib.load('model/le_house_type.pkl')
        return le_city, le_house_type
    except FileNotFoundError as e:
        raise Exception(f"LabelEncoder 檔案未找到：{e}")

def train_label_encoders(df):
    le_city       = LabelEncoder()
    le_house_type = LabelEncoder()

    # 若有空值或其他可能不在 df 的值，都先填成 'Unknown'
    df['City']      = df['City'].fillna('Unknown')
    df['HouseType'] = df['HouseType'].fillna('Unknown')

    # 在 fit 時，特別加入 'Unknown'
    le_city.fit(df['City'].tolist() + ['Unknown'])
    le_house_type.fit(df['HouseType'].tolist() + ['Unknown'])

    joblib.dump(le_city, 'model/le_city.pkl')
    joblib.dump(le_house_type, 'model/le_house_type.pkl')
    logging.info("LabelEncoders 已訓練並儲存。")
    return le_city, le_house_type

# -----------------------------------------------------------------------------
# 5. 資料預處理 & CP_value 計算
# -----------------------------------------------------------------------------
def extract_pattern_info(pattern, keyword):
    if isinstance(pattern, str):
        match = re.search(rf'(\d+){keyword}', pattern)
        return int(match.group(1)) if match else 0
    return 0

def preprocess_listing_for_cp_value(listing, le_city, le_house_type):
    """
    預處理房源資訊，並完成城市 / 房屋類型的編碼。
    若 City / HouseType 不在 LabelEncoder.classes_ 中，則一律視為 'Unknown'。
    若 Price 或 HouseType 缺失，也分別設為 0 / 'Unknown'。
    最後回傳一個包含必要特徵的字典 processed。
    """
    if not isinstance(listing, dict):
        raise ValueError("Listing should be a dictionary.")

    # -------- 1) Price / Housetype 基本欄位檢查 --------
    if 'Price' not in listing or listing['Price'] is None:
        listing['Price'] = 0
        logging.error(f"房源 ID {listing.get('Id','?')} 缺少 Price，設為 0。")

    if 'Housetype' not in listing or listing['Housetype'] is None:
        listing['Housetype'] = 'Unknown'

    # -------- 2) 準備城市 / 房屋類型 值，並檢查是否在 Encoder 裡 --------
    city_val  = listing.get('City', 'Unknown')
    house_val = listing.get('Housetype', 'Unknown')

    # 若 city_val 不在 le_city 的 classes_，改成 'Unknown'
    if city_val not in le_city.classes_:
        city_val = 'Unknown'

    # 若 house_val 不在 le_house_type 的 classes_，改成 'Unknown'
    if house_val not in le_house_type.classes_:
        house_val = 'Unknown'

    # -------- 3) 分別用 try-except 呼叫 transform，防止出錯 --------
    try:
        listing['City'] = le_city.transform([city_val])[0]
    except Exception as e:
        logging.warning(f"城市編碼失敗：{e}，強制設為 Unknown。")
        listing['City'] = le_city.transform(['Unknown'])[0]

    try:
        listing['HouseType'] = le_house_type.transform([house_val])[0]
    except Exception as e:
        logging.warning(f"房屋類型編碼失敗：{e}，強制設為 Unknown。")
        listing['HouseType'] = le_house_type.transform(['Unknown'])[0]

    # -------- 4) 處理 Pattern (Rooms, LivingRooms, Bathrooms, Balconies, Kitchens) --------
    listing['Rooms']       = extract_pattern_info(listing.get('Pattern',''), '房')
    listing['LivingRooms'] = extract_pattern_info(listing.get('Pattern',''), '廳')
    listing['Bathrooms']   = extract_pattern_info(listing.get('Pattern',''), '衛')
    listing['Balconies']   = extract_pattern_info(listing.get('Pattern',''), '陽台')
    listing['Kitchens']    = extract_pattern_info(listing.get('Pattern',''), '廚房')

    # -------- 5) 處理 Environment，拆成一維的 flat_env 列表 --------
    try:
        if isinstance(listing.get('Environment'), str):
            listing['Environment'] = json.loads(listing['Environment'])
        elif not isinstance(listing.get('Environment'), list):
            listing['Environment'] = []
    except Exception:
        listing['Environment'] = []

    flat_env = []
    for item in listing['Environment']:
        if isinstance(item, list):
            flat_env.extend(item)
        else:
            flat_env.append(item)

    # -------- 6) 加入預期的環境特徵 (Env_有對外窗, Env_有中庭, Env_樓中樓, Env_陽台, Env_為邊間) --------
    expected_env_features = ['Env_有對外窗','Env_有中庭','Env_樓中樓','Env_陽台','Env_為邊間']
    for feature in expected_env_features:
        listing[feature] = 1 if feature in flat_env else 0

    # -------- 7) 最後彙整需要的欄位，組合一個 processed dict --------
    necessary_cols = [
        'Size','Age','Bedroom','City','HouseType','Rooms','LivingRooms',
        'Bathrooms','Balconies','Kitchens'
    ]
    necessary_cols.extend(expected_env_features)

    processed = {key: listing.get(key, 0) for key in necessary_cols}
    processed['Price'] = listing.get('Price', 0)

    return processed


def compute_cp_value(total_facilities, price):
    if price == 0:
        return 0
    return (total_facilities / price)*10000

# -----------------------------------------------------------------------------
# 6. 從資料庫讀取、更新
# -----------------------------------------------------------------------------
def load_house_data(db_path):
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT 
            id, title AS Name, price AS Price, size AS Size, age AS Age, floors AS Floors, bedroom AS Bedroom,
            city AS City, location AS Location, house_type AS HouseType, pattern AS Pattern,
            environment AS Environment, url AS Url, image AS Image_name
        FROM listings
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logging.error(f"載入房屋資料失敗：{e}")
        raise

def update_cp_values_in_db(db_path, processed_df):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for idx, row in processed_df.iterrows():
            listing_id = row.get('id')
            cp_value   = row.get('CP_value', 0)
            if listing_id is not None:
                cursor.execute("UPDATE listings SET CP_value=? WHERE id=?", (cp_value, listing_id))
        conn.commit()
        conn.close()
        logging.info("已將 CP_value 寫回資料庫。")
    except Exception as e:
        logging.error(f"更新 CP_value 到資料庫時發生錯誤：{e}")

# -----------------------------------------------------------------------------
# 7. 其他功能
# -----------------------------------------------------------------------------
def knn_similar_listings(listings, target_listing, n_neighbors=3):
    city = target_listing['city']
    same_city = [l for l in listings if l['city'] == city]
    if len(same_city) < 1: 
        return []

    n_neighbors = min(n_neighbors, len(same_city), 3)
    features    = ['price','bedroom','age','floors']
    data_matrix = np.array([[l[f] for f in features] for l in same_city])
    target_feat = np.array([target_listing[f] for f in features]).reshape(1,-1)

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(data_matrix)
    _, indices = knn.kneighbors(target_feat)
    return [same_city[idx] for idx in indices[0]]

def preprocess_coordinates(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coordinates (
            listing_id INTEGER PRIMARY KEY,
            latitude   REAL,
            longitude  REAL
        )
    """)
    conn.commit()

    cursor.execute("""
        SELECT id, location FROM listings
        WHERE id NOT IN (SELECT listing_id FROM coordinates)
    """)
    geolocator = Nominatim(user_agent="my_geo_app")
    for listing_id, address in cursor.fetchall():
        lat, lng = get_coordinates(address, geolocator)
        cursor.execute("""
            INSERT INTO coordinates (listing_id, latitude, longitude)
            VALUES (?,?,?)
        """,(listing_id, lat, lng))
        conn.commit()
        logging.info(f"已編碼地址 ID {listing_id}")
        sleep(1)
    conn.close()
    
def fix_location(address):
    """
    若地址中含有 '/', 則只取第一段，忽略 '/' 後面的文字。
    """
    if '/' in address:
        # split('/', 1) -> 最多分割一次；只取前半部分
        return address.split('/', 1)[0].strip()
    return address.strip()

def init_label_encoders(df):
    """
    若 'model/le_city.pkl'、'model/le_house_type.pkl' 存在，則直接 load；
    否則用 df 訓練後再存檔，最後回傳 le_city, le_house_type。
    """
    import os

    city_path  = 'model/le_city.pkl'
    house_path = 'model/le_house_type.pkl'

    # 若檔案都存在，直接 load
    if os.path.exists(city_path) and os.path.exists(house_path):
        try:
            le_city       = joblib.load(city_path)
            le_house_type = joblib.load(house_path)
            return le_city, le_house_type
        except:
            logging.warning("LabelEncoder 檔案載入失敗，改用 df 訓練並存檔")

    # 若檔案不存在或載入失敗，就用 df 訓練
    le_city       = LabelEncoder()
    le_house_type = LabelEncoder()

    df['City']      = df['City'].fillna('Unknown')
    df['HouseType'] = df['HouseType'].fillna('Unknown')

    le_city.fit(df['City'].tolist() + ['Unknown'])
    le_house_type.fit(df['HouseType'].tolist() + ['Unknown'])

    joblib.dump(le_city, city_path)
    joblib.dump(le_house_type, house_path)

    logging.info("LabelEncoders 已重新訓練並儲存。")
    return le_city, le_house_type


# -----------------------------------------------------------------------------
# 8. 模型訓練主函數 (train_xgboost)
# -----------------------------------------------------------------------------
def train_xgboost(db_path, model_path='model/trained_model.xgb'):
    logging.info("開始讀取房屋資料...")
    df = load_house_data(db_path)
    df = load_house_data(db_path)

    logging.info("開始訓練 / 讀取 LabelEncoder...")
    le_city, le_house_type = train_label_encoders(df)  # 如果沒先訓練，可改 train_label_encoders(df)

    geolocator = Nominatim(user_agent="my_geo_app")

    def process_listing(row):
        listing = dict(row)
        address = listing.get("Location","")
        # 1. 取得周邊便利度
        conv_feats = get_convenience_features(address, geolocator)

        # 2. 預處理 (LabelEncoder 等)
        processed = preprocess_listing_for_cp_value(listing, le_city, le_house_type)

        # 3. 計算 CP_value
        cpv = compute_cp_value(conv_feats.get("total_facilities",0), processed["Price"])
        processed["CP_value"] = cpv

        # 4. 加入 id (方便後續UPDATE)
        processed["id"] = listing.get("id")

        return processed

    logging.info("開始並行處理房源，計算 CP_value...")
    processed_list = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_listing, row) for _,row in df.iterrows()]
        for future in as_completed(futures):
            try:
                processed = future.result()
                processed_list.append(processed)
            except Exception as e:
                logging.error(f"處理房源失敗：{e}")

    processed_df = pd.DataFrame(processed_list)

    # 寫回資料庫 CP_value
    update_cp_values_in_db(db_path, processed_df)

    # ====== 準備特徵與目標 ======
    feature_cols = [
        "Size","Age","Bedroom","City","HouseType","Rooms","LivingRooms","Bathrooms",
        "Balconies","Kitchens","Env_有對外窗","Env_有中庭","Env_樓中樓","Env_陽台","Env_為邊間"
        # 如果不需要 'total_facilities' 就省略
    ]

    if not set(feature_cols).issubset(processed_df.columns):
        missing_cols = set(feature_cols) - set(processed_df.columns)
        logging.error(f"以下特徵欄位缺失：{missing_cols}")
        # 這裡您可選擇 raise Exception or 自動補 0
        for col in missing_cols:
            processed_df[col] = 0

    X = processed_df[feature_cols].fillna(0)
    y = processed_df["CP_value"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [3, 5],
        "learning_rate":[0.01,0.1],
        "subsample":    [0.8,1.0]
    }

    logging.info("開始 GridSearch 訓練模型...")
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=3, scoring="neg_mean_squared_error",
        verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"Best params: {grid_search.best_params_}")
    logging.info(f"Best score (MSE): {-grid_search.best_score_}")

    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, model_path)
    logging.info(f"模型已儲存到 {model_path}")

    train_pred = best_model.predict(X_train)
    test_pred  = best_model.predict(X_test)
    logging.info(f"Train MSE: {mean_squared_error(y_train,train_pred)}")
    logging.info(f"Test MSE: {mean_squared_error(y_test,test_pred)}")
    logging.info(f"Train R2: {r2_score(y_train,train_pred)}")
    logging.info(f"Test R2: {r2_score(y_test,test_pred)}")

# -----------------------------------------------------------------------------
# 9. 預測函數
# -----------------------------------------------------------------------------
def predict_cp_value(listing, model_path='model/trained_model.xgb',
                     le_city_path='model/le_city.pkl',
                     le_house_type_path='model/le_house_type.pkl'):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise Exception("模型檔案未找到，請先訓練並儲存 'trained_model.xgb'。")

    try:
        le_city        = joblib.load(le_city_path)
        le_house_type  = joblib.load(le_house_type_path)
    except FileNotFoundError as e:
        raise Exception(f"LabelEncoder 檔案未找到：{e}")

    try:
        processed = preprocess_listing_for_cp_value(listing, le_city, le_house_type)
    except ValueError as ve:
        logging.error(f"預處理房源失敗：{ve}")
        return None

    df = pd.DataFrame([processed])
    try:
        pred = model.predict(df)
    except Exception as e:
        logging.error(f"模型預測失敗：{e}")
        return None
    return pred

# -----------------------------------------------------------------------------
# 10. 主程式入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'database.db')
    logging.info("開始預先地理編碼所有地址...")
    preprocess_coordinates(db_path)
    logging.info("地理編碼完成。")

    logging.info("開始訓練 XGBoost 模型...")
    train_xgboost(db_path)
    logging.info("模型訓練完成。")
