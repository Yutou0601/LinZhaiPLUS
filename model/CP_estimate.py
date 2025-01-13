##############################################
# model/CP_estimate.py
##############################################
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

# ===== 全局緩存字典 (用於地理編碼 & 便利度查詢快取) =====
address_cache = {}

# -----------------------------------------------------------------------------
# 1. 座標與地址處理
# -----------------------------------------------------------------------------

def fix_location(address):
    """
    若地址中含有 '/', 則只取第一段，忽略後面部分
    """
    if '/' in address:
        return address.split('/', 1)[0].strip()
    return address.strip()

def get_coordinates(address, geolocator, timeout=10):
    """
    透過 Nominatim (geopy) 取得地址座標，並快取於 address_cache
    """
    address = fix_location(address)
    formatted_address = address.replace('/', ' ').strip()

    # 若已在快取中, 直接回傳
    if formatted_address in address_cache:
        cached = address_cache[formatted_address]
        return cached.get('latitude'), cached.get('longitude')

    # 否則嘗試 geolocator
    try:
        loc = geolocator.geocode(formatted_address, timeout=timeout)
        if loc:
            lat, lng = loc.latitude, loc.longitude
            address_cache[formatted_address] = {'latitude': lat, 'longitude': lng}
            return lat, lng
        else:
            logging.warning(f"無法定位地址: {address}")
            address_cache[formatted_address] = {'latitude': None, 'longitude': None}
            return None, None
    except Exception as e:
        logging.warning(f"定位地址失敗: {address}, 錯誤: {e}")
        address_cache[formatted_address] = {'latitude': None, 'longitude': None}
        return None, None

# -----------------------------------------------------------------------------
# 2. 使用 Overpass API 搜尋附近設施
# -----------------------------------------------------------------------------

def search_osm_data(lat, lng, category, radius=1000, max_retries=3):
    """
    向 Overpass API 查詢指定 category 的 POI
    """
    # 台灣範圍檢查
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
                logging.error(f"OSM API 請求失敗: {response.status_code}")
        except requests.exceptions.Timeout:
            logging.error("OSM API 請求逾時")
        except Exception as e:
            logging.error(f"OSM API 請求錯誤: {e}")
        attempt += 1
        sleep(2)

    logging.error("超過最大重試次數，無法取得資料")
    return []

def _extract_locations(elements):
    """
    從 Overpass 回傳的 elements 中提取 (lat, lng)
    """
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
# 3. 便利度計算 (周邊設施)
# -----------------------------------------------------------------------------

def _zero_facilities():
    """
    若定位失敗, 回傳全 0
    """
    return {
        "amenity_count":0,"shop_count":0,"leisure_count":0,"tourism_count":0,"education_count":0,
        "healthcare_count":0,"public_transport_count":0,"restaurant_count":0,"fast_food_count":0,
        "cafe_count":0,"tram_stop_count":0,"subway_entrance_count":0,"bus_stop_count":0,
        "train_station_count":0,"parking_count":0,"total_facilities":0
    }

def _update_address_cache(address, facilities):
    """
    更新 address_cache 中的 'facilities' 欄位
    """
    if address in address_cache:
        address_cache[address]['facilities'] = facilities
    else:
        address_cache[address] = {'facilities': facilities}

def get_convenience_features(address, geolocator):
    """
    回傳當前地址附近的各類設施數量 dict,
    並以 address_cache 加速重複查詢
    """
    # 若 cache 中已有, 直接使用
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
        locs = search_osm_data(lat,lng, category, radius=1000)
        convenience_dict[f"{category}_count"] = len(locs)
        total_count += len(locs)

    convenience_dict["total_facilities"] = total_count
    _update_address_cache(address, convenience_dict)
    return convenience_dict

# -----------------------------------------------------------------------------
# 4. LabelEncoder 相關
# -----------------------------------------------------------------------------
def load_label_encoders():
    """
    載入已訓練好的 LabelEncoders (le_city.pkl, le_house_type.pkl)
    """
    try:
        le_city = joblib.load('model/le_city.pkl')
        le_house_type = joblib.load('model/le_house_type.pkl')
        return le_city, le_house_type
    except FileNotFoundError as e:
        raise Exception(f"LabelEncoder 檔案未找到: {e}")

def train_label_encoders(df):
    """
    針對 City, HouseType 訓練並儲存 LabelEncoder
    """
    from sklearn.preprocessing import LabelEncoder
    le_city = LabelEncoder()
    le_house_type = LabelEncoder()

    df['City'] = df['City'].fillna('Unknown')
    df['HouseType'] = df['HouseType'].fillna('Unknown')

    le_city.fit(df['City'].tolist() + ['Unknown'])
    le_house_type.fit(df['HouseType'].tolist() + ['Unknown'])

    joblib.dump(le_city,'model/le_city.pkl')
    joblib.dump(le_house_type,'model/le_house_type.pkl')
    logging.info("LabelEncoders 已重新訓練並儲存.")
    return le_city, le_house_type

# -----------------------------------------------------------------------------
# 5. 資料預處理 & CP_value 計算
# -----------------------------------------------------------------------------

def extract_pattern_info(pattern, keyword):
    """
    從 pattern (e.g. '3房2廳2衛') 中提取指定keyword數量
    """
    if isinstance(pattern, str):
        m = re.search(rf'(\d+){keyword}', pattern)
        return int(m.group(1)) if m else 0
    return 0

def preprocess_listing_for_cp_value(listing, le_city, le_house_type):
    """
    把房源 listing 做以下動作:
    1) 檢查 Price/Housetype
    2) LabelEncoder City/Housetype
    3) 解析 pattern => Rooms, LivingRooms, ...
    4) 解析 environment => Env_有對外窗, Env_有中庭 ...
    5) 回傳訓練所需欄位
    """
    if not isinstance(listing, dict):
        raise ValueError("Listing should be a dictionary.")

    # Price
    if 'Price' not in listing or listing['Price'] is None:
        listing['Price'] = 0
        logging.error(f"房源 ID {listing.get('Id','?')} 缺少 Price, 設為 0")

    # Housetype
    if 'Housetype' not in listing or listing['Housetype'] is None:
        listing['Housetype'] = 'Unknown'

    # 驗證 city/housetype 是否在 encoder
    city_val = listing.get('City','Unknown')
    house_val= listing.get('Housetype','Unknown')

    if city_val not in le_city.classes_:
        city_val='Unknown'
    if house_val not in le_house_type.classes_:
        house_val='Unknown'

    try:
        listing['City']= le_city.transform([city_val])[0]
    except:
        listing['City']= le_city.transform(['Unknown'])[0]

    try:
        listing['HouseType']= le_house_type.transform([house_val])[0]
    except:
        listing['HouseType']= le_house_type.transform(['Unknown'])[0]

    # 解析格局
    listing['Rooms'] = extract_pattern_info(listing.get('Pattern',''),'房')
    listing['LivingRooms'] = extract_pattern_info(listing.get('Pattern',''),'廳')
    listing['Bathrooms'] = extract_pattern_info(listing.get('Pattern',''),'衛')
    listing['Balconies'] = extract_pattern_info(listing.get('Pattern',''),'陽台')
    listing['Kitchens'] = extract_pattern_info(listing.get('Pattern',''),'廚房')

    # Environment
    try:
        if isinstance(listing.get('Environment'), str):
            listing['Environment']= json.loads(listing['Environment'])
        elif not isinstance(listing['Environment'], list):
            listing['Environment']=[]
    except:
        listing['Environment']=[]

    flat_env=[]
    for e in listing['Environment']:
        if isinstance(e, list):
            flat_env.extend(e)
        else:
            flat_env.append(e)

    env_features = ['Env_有對外窗','Env_有中庭','Env_樓中樓','Env_陽台','Env_為邊間']
    for f in env_features:
        listing[f]= 1 if f in flat_env else 0

    needed_cols = [
        'Size','Age','Bedroom','City','HouseType','Rooms','LivingRooms',
        'Bathrooms','Balconies','Kitchens'
    ] + env_features
    processed = {k: listing.get(k,0) for k in needed_cols}
    processed['Price']= listing.get('Price',0)
    return processed

def compute_cp_value(total_facilities, price):
    """
    CP_value = (便利度 total_facilities / Price) * 10000
    """
    if price==0:
        return 0
    return (total_facilities / price)*10000

# -----------------------------------------------------------------------------
# 6. 從資料庫讀取/寫回 CP_value
# -----------------------------------------------------------------------------
def load_house_data(db_path):
    """
    讀取 listings 表, 回傳 DataFrame
    """
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
        logging.error(f"載入房屋資料失敗: {e}")
        raise

def update_cp_values_in_db(db_path, processed_df):
    """
    將計算好的 CP_value 寫回 listings 表
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for idx, row in processed_df.iterrows():
            listing_id = row.get('id')
            cp_value = row.get('CP_value', 0)
            if listing_id is not None:
                cursor.execute("UPDATE listings SET CP_value=? WHERE id=?", (cp_value, listing_id))
        conn.commit()
        conn.close()
        logging.info("已將 CP_value 寫回資料庫。")
    except Exception as e:
        logging.error(f"更新 CP_value 到資料庫時發生錯誤: {e}")

# -----------------------------------------------------------------------------
# 7. 其他功能
# -----------------------------------------------------------------------------
def knn_similar_listings(listings, target_listing, n_neighbors=3):
    """
    KNN 推薦: 先篩同城市，再比較 (price, bedroom, age, floors)
    """
    city = target_listing.get('city')
    same_city = [l for l in listings if l['city'] == city]
    if len(same_city) < 1:
        return []

    n_neighbors = min(n_neighbors, len(same_city), 3)
    features = ['price','bedroom','age','floors']
    data_matrix = np.array([[l[f] for f in features] for l in same_city])
    target_feat = np.array([target_listing[f] for f in features]).reshape(1,-1)

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(data_matrix)
    _, indices = knn.kneighbors(target_feat)
    return [same_city[idx] for idx in indices[0]]

def preprocess_coordinates(db_path):
    """
    可選: 先將 listings 的地址做 geocode, 寫入 coordinates 表
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coordinates (
            listing_id INTEGER PRIMARY KEY,
            latitude REAL,
            longitude REAL
        )
    """)
    conn.commit()

    cursor.execute("""
        SELECT id, location FROM listings
        WHERE id NOT IN (SELECT listing_id FROM coordinates)
    """)
    geolocator = Nominatim(user_agent="my_geo_app")
    for listing_id, address in cursor.fetchall():
        lat,lng = get_coordinates(address, geolocator)
        cursor.execute("""
            INSERT INTO coordinates(listing_id, latitude, longitude)
            VALUES (?,?,?)
        """, (listing_id, lat, lng))
        conn.commit()
        logging.info(f"已編碼地址 ID {listing_id}")
        sleep(1)
    conn.close()

# -----------------------------------------------------------------------------
# 8. 模型訓練主函數 (train_xgboost)
# -----------------------------------------------------------------------------
def train_xgboost(db_path, model_path='model/trained_model.xgb'):
    """
    主入口:
    1. 讀取 DB listings
    2. LabelEncoder 訓練/讀取
    3. 分批計算周邊便利度 -> CP_value
    4. 更新 CP_value 回 DB
    5. 以 XGBoost 進行訓練
    """
    logging.info("開始讀取房屋資料...")
    df = load_house_data(db_path)

    logging.info("開始訓練 / 讀取 LabelEncoder...")
    le_city, le_house_type = train_label_encoders(df)

    geolocator = Nominatim(user_agent="my_geo_app")

    def process_listing(row):
        listing = dict(row)
        address = listing.get("Location", "")
        # (1) 計算周邊便利度
        conv_feats = get_convenience_features(address, geolocator)
        # (2) 預處理 (LabelEncoder 等)
        processed = preprocess_listing_for_cp_value(listing, le_city, le_house_type)
        # (3) 計算 CP_value
        cpv = compute_cp_value(conv_feats.get("total_facilities",0), processed["Price"])
        processed["CP_value"] = cpv
        # (4) 加入 id
        processed["id"] = listing.get("id")
        return processed

    logging.info("開始並行處理房源, 計算 CP_value...")
    processed_list = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_listing, row) for _, row in df.iterrows()]
        for fut in as_completed(futures):
            try:
                processed_list.append(fut.result())
            except Exception as e:
                logging.error(f"處理房源失敗: {e}")

    processed_df = pd.DataFrame(processed_list)

    # 寫回 DB
    update_cp_values_in_db(db_path, processed_df)

    # ====== 訓練 XGBoost ======
    feature_cols = [
        "Size","Age","Bedroom","City","HouseType","Rooms","LivingRooms","Bathrooms",
        "Balconies","Kitchens","Env_有對外窗","Env_有中庭","Env_樓中樓","Env_陽台","Env_為邊間"
    ]
    missing_cols = set(feature_cols) - set(processed_df.columns)
    if missing_cols:
        logging.error(f"以下特徵欄位缺失: {missing_cols}")
        for col in missing_cols:
            processed_df[col] = 0

    X = processed_df[feature_cols].fillna(0)
    y = processed_df["CP_value"].fillna(0)

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    }

    logging.info("開始 GridSearch 訓練模型...")
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"Best params: {grid_search.best_params_}")
    logging.info(f"Best score (MSE): {-grid_search.best_score_}")

    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, model_path)
    logging.info(f"模型已儲存到 {model_path}")

    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    logging.info(f"Train MSE: {mean_squared_error(y_train, train_pred)}")
    logging.info(f"Test MSE: {mean_squared_error(y_test, test_pred)}")
    logging.info(f"Train R2: {r2_score(y_train, train_pred)}")
    logging.info(f"Test R2: {r2_score(y_test, test_pred)}")

# -----------------------------------------------------------------------------
# 9. 預測函數
# -----------------------------------------------------------------------------
def predict_cp_value(listing, model_path='model/trained_model.xgb',
                     le_city_path='model/le_city.pkl',
                     le_house_type_path='model/le_house_type.pkl'):
    """
    讀取 XGBoost 模型 & LabelEncoders, 預測單筆 listing 的 CP_value
    """
    try:
        model = joblib.load(model_path)
        le_city = joblib.load(le_city_path)
        le_house_type = joblib.load(le_house_type_path)
    except FileNotFoundError:
        logging.error("找不到模型/LabelEncoder檔案, 請先訓練!")
        return None

    # 預處理
    try:
        processed = preprocess_listing_for_cp_value(listing, le_city, le_house_type)
    except Exception as e:
        logging.error(f"預處理失敗: {e}")
        return None

    df = pd.DataFrame([processed])
    try:
        pred = model.predict(df)
        return pred
    except Exception as e:
        logging.error(f"模型預測失敗: {e}")
        return None

# -----------------------------------------------------------------------------
# 10. 主程式入口
# -----------------------------------------------------------------------------
if __name__=="__main__":
    db_path = os.path.join(os.path.dirname(__file__), '..','data','database.db')
    logging.info("開始預先地理編碼...")
    preprocess_coordinates(db_path)
    logging.info("地理編碼完成.開始訓練/更新 CP_value...")
    train_xgboost(db_path)
    logging.info("CP_value 已更新 & 模型完成.")
