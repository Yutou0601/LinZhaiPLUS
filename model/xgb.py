import pandas as pd
import joblib
import re
import json
import logging
from sklearn.preprocessing import LabelEncoder

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 載入已訓練的模型 (使用 XGBRegressor)
try:
    model = joblib.load("model/trained_model.xgb")
    le_city = joblib.load('model/le_city.pkl')
    le_house_type = joblib.load('model/le_house_type.pkl')
    logging.info("模型和 LabelEncoder 載入成功。")
except FileNotFoundError as e:
    logging.error(f"載入模型或 LabelEncoder 失敗: {e}")
    raise

def extract_pattern_info(pattern, keyword):
    if isinstance(pattern, str):
        match = re.search(rf'(\d+){keyword}', pattern)
        return int(match.group(1)) if match else 0
    return 0

def preprocess_listing(listing):
    """預處理房屋資料，讓 XGBoost 能正確進行預測。"""
    if not isinstance(listing, dict):
        raise ValueError("Listing should be a dictionary.")
    
    # 將所有欄位名稱轉換為首字母大寫，並處理底線。例如 city -> City, house_type -> HouseType
    listing = {key.replace('_', '').capitalize(): value for key, value in listing.items()}

    # 確保 'HouseType' 欄位存在
    if 'Housetype' not in listing:
        raise ValueError("'HouseType' not found in listing data.")

    # -----------------------------------------------------
    # 在 transform 之前，若發現 City or HouseType == '0'，
    # 則改為 'Unknown'。也可檢查空字串或整數 0。
    # -----------------------------------------------------
    city_val = str(listing.get('City', '')).strip()
    if city_val in ['0', '']:
        city_val = 'Unknown'

    house_val = str(listing.get('Housetype', '')).strip()
    if house_val in ['0', '']:
        house_val = 'Unknown'

    # -------------------------
    # LabelEncoder transform
    # -------------------------
    try:
        listing['City'] = le_city.transform([city_val])[0]
    except Exception as e:
        logging.warning(f"City '{city_val}' 未見於訓練資料，設為 'Unknown'。")
        listing['City'] = le_city.transform(['Unknown'])[0]
    
    try:
        listing['HouseType'] = le_house_type.transform([house_val])[0]
    except Exception as e:
        logging.warning(f"HouseType '{house_val}' 未見於訓練資料，設為 'Unknown'。")
        listing['HouseType'] = le_house_type.transform(['Unknown'])[0]

    # 處理 pattern 與 environment 欄位
    listing['Rooms'] = extract_pattern_info(listing.get('Pattern', ''), '房')
    listing['LivingRooms'] = extract_pattern_info(listing.get('Pattern', ''), '廳')
    listing['Bathrooms'] = extract_pattern_info(listing.get('Pattern', ''), '衛')
    listing['Balconies'] = extract_pattern_info(listing.get('Pattern', ''), '陽台')
    listing['Kitchens'] = extract_pattern_info(listing.get('Pattern', ''), '廚房')

    # 確保環境資料是扁平的
    try:
        if isinstance(listing.get('Environment'), str):
            listing['Environment'] = json.loads(listing['Environment'])
        elif not isinstance(listing.get('Environment'), list):
            listing['Environment'] = []
    except json.JSONDecodeError:
        logging.warning("Environment 欄位 JSON 解碼失敗，設為空列表。")
        listing['Environment'] = []

    flat_environment = []
    for sublist in listing['Environment']:
        if isinstance(sublist, list):
            flat_environment.extend(sublist)
        else:
            flat_environment.append(sublist)

    # 定義期望的環境特徵
    expected_env_features = ['Env_有對外窗','Env_有中庭','Env_樓中樓','Env_陽台','Env_為邊間']
    for feature in expected_env_features:
        listing[feature] = 1 if feature in flat_environment else 0

    necessary_columns = [
        'Size','Age','Bedroom','City','HouseType','Rooms','LivingRooms',
        'Bathrooms','Balconies','Kitchens'
    ]
    necessary_columns.extend(expected_env_features)

    # 最後只取出必要欄位
    processed = {key: listing.get(key, 0) for key in necessary_columns}
    return processed

def predict_cp_value(listing):
    """呼叫此函式進行 CP_value 預測。"""
    try:
        processed_listing = preprocess_listing(listing)
    except Exception as e:
        logging.error(f"預處理失敗: {e}")
        return None
    
    # 轉成 DataFrame
    data = pd.DataFrame([processed_listing])

    try:
        # 預測結果
        predicted_cp_value = model.predict(data)
        return predicted_cp_value
    except Exception as e:
        logging.error(f"模型預測失敗: {e}")
        return None
