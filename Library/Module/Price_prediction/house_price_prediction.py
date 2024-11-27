# house_price_prediction_with_li.py

import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 1. 載入並預處理資料集
current_dir = os.path.dirname(os.path.abspath(__file__))
house_rent_csv_path = os.path.join(current_dir, 'House_Rent_Info.csv')
weight_csv_path = os.path.join(current_dir, 'Taipei_town_weight.csv')
li_weight_csv_path = os.path.join(current_dir, 'Taipei_Li_weight.csv')  # 新增

# 讀取主要資料集
data = pd.read_csv(house_rent_csv_path)

# 顯示資料集的前幾行和欄位名稱以進行調試
print("House_Rent_Info.csv 的前幾行：")
print(data.head())
print("\nHouse_Rent_Info.csv 的欄位名稱：")
print(data.columns.tolist())

# 2. 檢查 'District' 和 'District_Index' 是否存在，並去除多餘空格
data.columns = data.columns.str.strip()
if 'District' not in data.columns or 'District_Index' not in data.columns:
    raise KeyError("CSV 檔案中缺少 'District' 或 'District_Index' 欄位。請確認檔案格式正確。")

# 3. 確保重新命名成功
data.rename(columns={'District': '區', 'District_Index': '區_編號'}, inplace=True)

# 驗證欄位是否已正確重新命名
print("\n重新命名後的欄位名稱：")
print(data.columns.tolist())

if '區' not in data.columns or '區_編號' not in data.columns:
    raise KeyError("重新命名後的欄位 '區' 或 '區_編號' 不存在。請確認重新命名步驟正確。")

# 4. 排除 '區_編號' 為 0 的資料（未知區域）
initial_row_count = data.shape[0]
data = data[data['區_編號'] != 0]
cleaned_row_count = data.shape[0]
rows_dropped = initial_row_count - cleaned_row_count
print(f"\n已排除 {rows_dropped} 筆未知區域的資料。")

# 5. 標準化 '縣市' 名稱：將 '臺' 替換為 '台' 並移除所有空格
data['縣市'] = data['Location'].str.replace('臺', '台').str.replace(' ', '')
data['區'] = data['區'].str.replace(' ', '')  # 移除 '區' 欄位中的空格
print("已標準化 '縣市' 和 '區' 名稱。")

# 6. 顯示更新後的資料集前幾行以確認
print("\n更新後資料集的前幾行：")
print(data.head())

# 7. 載入台北市各區域的權重資料
weight_data = pd.read_csv(weight_csv_path)

# 顯示權重資料集的前幾行和欄位名稱以進行調試
print("\nTaipei_town_weight.csv 的前幾行：")
print(weight_data.head())
print("\nTaipei_town_weight.csv 的欄位名稱：")
print(weight_data.columns.tolist())

# 8. 移除 '總計' 行，因為它不對應任何具體區域
weight_data = weight_data[weight_data['區'] != '總計']
print("\n移除 '總計' 行後的 Taipei_town_weight.csv：")
print(weight_data.head())

# 確保 '縣市' 和 '區' 欄位的名稱一致，以便合併
if '縣市' not in weight_data.columns or '區' not in weight_data.columns:
    raise ValueError("權重資料集中必須包含 '縣市' 和 '區' 欄位。")

# 9. 合併主資料集與權重資料集
data = pd.merge(data, weight_data, on=['縣市', '區'], how='left')

# 顯示合併後的資料集前幾行以確認合併是否成功
print("\n合併後資料集的前幾行：")
print(data.head())

# 10. 處理缺失的權重值和顏色值
data['權重'] = data['權重'].fillna(data['權重'].median())
data['顏色'] = data['顏色'].fillna('無色')
print("已填補缺失的 '權重' 和 '顏色' 值。")

# 11. 將 '顏色' 映射到數值，以反映其優先級
color_mapping = {
    '紅色': 500,   # 核心市區或高密度商業區
    '橙色': 400,   # 次市區
    '黃色': 300,   # 人口密度適中，基礎設施完善
    '綠色': 200,   # 普通住宅區
    '無色': 100    # 偏遠但有一定基礎設施
}
data['顏色_數值'] = data['顏色'].map(color_mapping).fillna(0).astype(int)
print("已將 '顏色' 映射為數值。")

# 12. 處理 'Tags' 欄位，提取關鍵字並創建二元特徵
# 提取所有唯一的標籤
unique_tags = set()

for tags_str in data['Tags'].dropna():
    tags_list = [tag.strip() for tag in tags_str.split(',')]
    unique_tags.update(tags_list)

# 建立中文標籤到英文標籤的對照表
tag_translation = {
    '景觀宅': 'Scenic_House',
    '邊間': 'Corner_Unit',
    '房間皆有窗': 'All_Rooms_with_Windows',
    '近公車站': 'Near_Bus_Stop',
    '高樓層': 'High_Floor',
    '近公園': 'Near_Park',
    '頂樓': 'Top_Floor',
    '格局方正': 'Regular_Layout',
    '免爬樓梯': 'No_Stairs_Needed',
    '獨棟': 'Detached_House',
    '近商圈': 'Near_Commercial_Area',
    '次頂樓': 'Second_Top_Floor',
    '重劃區': 'Redevelopment_Area',
    '無暗房': 'No_Dark_Rooms',
    '雙衛浴': 'Two_Bathrooms',
    '近捷運': 'Near_MRT',
    '雙車位': 'Double_Parking_Space',
    '平面車位': 'Ground_Parking',
    '有露臺': 'Has_Terrace',
    '近台鐵': 'Near_Taiwan_Railway',
    '大面寬': 'Wide_Frontage',
    '宜收租': 'Good_for_Rental',
    '近學區': 'Near_School_District',
    '廁所開窗': 'Bathroom_Window',
    '一層一戶': 'One_Unit_Per_Floor',
    '有庭院': 'Has_Courtyard',
    '三面採光': 'Three-sided_Lighting',
    '永久棟距': 'Permanent_Building_Distance',
    '採光佳': 'Good_Lighting',
    '裝潢美宅': 'Beautifully_Decorated_House',
    '前後陽台': 'Front_and_Rear_Balconies',
    # 如有其他標籤，請在此添加
}

# 將中文標籤轉換為英文標籤
data['Tags_English'] = data['Tags'].apply(
    lambda x: ','.join([tag_translation.get(tag.strip(), tag.strip()) for tag in x.split(',')]) if isinstance(x, str) else x
)

# 更新 unique_tags 為英文標籤
unique_tags_english = set()

for tags_str in data['Tags_English'].dropna():
    tags_list = [tag.strip() for tag in tags_str.split(',')]
    unique_tags_english.update(tags_list)

# 計算每個標籤的出現次數
tag_counts = {}
for tag in unique_tags_english:
    count = data['Tags_English'].apply(lambda x: tag in x if isinstance(x, str) else False).sum()
    tag_counts[tag] = count

# 設定出現次數的閾值，僅保留高頻標籤
threshold = 30  # 根據實際情況調整
selected_tags = [tag for tag, count in tag_counts.items() if count >= threshold]

print(f"\n選定的高頻標籤數量：{len(selected_tags)}")
print(f"選定的高頻標籤：{selected_tags}")

# 為每個選定的標籤創建一個新的二元特徵欄位
for tag in selected_tags:
    col_name = 'tag_' + tag.replace(' ', '_')
    data[col_name] = data['Tags_English'].apply(lambda x: int(tag in x) if isinstance(x, str) else 0)

# 13. 刪除不需要的欄位
data = data.drop(['ID', 'Name', 'Location', 'Tags', 'Tags_English', '顏色'], axis=1)
print("已刪除不需要的欄位。")

# 14. 處理數值型特徵
# 確保 'Floor' 和 'Age' 欄位為數值型
data['Floor'] = pd.to_numeric(data['Floor'], errors='coerce')
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

# 針對性地處理缺失值
data['Floor'] = data['Floor'].fillna(data['Floor'].median())
data['Age'] = data['Age'].fillna(data['Age'].median())

# 去除價格的異常值
# 假設合理的價格範圍為 50 萬到 1.5 億
data = data[(data['Price'] >= 500000) & (data['Price'] <= 150000000)]
print("已去除價格的異常值。")

# 對價格進行對數變換
data['Price'] = np.log1p(data['Price'])
print("已對價格進行對數變換。")

# 15. 載入台北市各里的權重資料
li_weight_data = pd.read_csv(li_weight_csv_path)

# 顯示權重資料集的前幾行和欄位名稱以進行調試
print("\nTaipei_Li_weight.csv 的前幾行：")
print(li_weight_data.head())
print("\nTaipei_Li_weight.csv 的欄位名稱：")
print(li_weight_data.columns.tolist())

# 標準化 '縣市' 和 '區' 名稱
li_weight_data['縣市'] = li_weight_data['縣市'].str.replace('臺', '台').str.replace(' ', '')
li_weight_data['區'] = li_weight_data['區'].str.replace(' ', '')
li_weight_data['里'] = li_weight_data['里'].str.replace(' ', '')

# 16. 計算各區符合區顏色的里的權重統計量
# 先建立區到顏色的映射
district_color_mapping = weight_data.set_index(['縣市', '區'])['顏色'].to_dict()

# 將 '顏色' 映射到數值，以反映其優先級
li_weight_data['顏色_數值'] = li_weight_data['顏色'].map(color_mapping).fillna(0).astype(int)

# 為了進行合併，添加 '區_顏色_數值' 到 li_weight_data
li_weight_data['區_顏色_數值'] = li_weight_data.apply(
    lambda x: color_mapping.get(district_color_mapping.get((x['縣市'], x['區']), '無色'), 0), axis=1)

# 選擇與區顏色匹配的里
li_weight_data_matched = li_weight_data[li_weight_data['顏色_數值'] == li_weight_data['區_顏色_數值']]

# 計算各區的里權重統計量
li_stats = li_weight_data_matched.groupby(['縣市', '區']).agg(
    Li_權重_mean=('權重', 'mean'),
    Li_權重_median=('權重', 'median'),
    Li_權重_std=('權重', 'std'),
    Li_count=('權重', 'count')
).reset_index()

# 將統計量合併到主資料集
data = pd.merge(data, li_stats, on=['縣市', '區'], how='left')

# 填補可能的缺失值
data['Li_權重_mean'] = data['Li_權重_mean'].fillna(data['Li_權重_mean'].median())
data['Li_權重_median'] = data['Li_權重_median'].fillna(data['Li_權重_median'].median())
data['Li_權重_std'] = data['Li_權重_std'].fillna(0)  # 標準差為 0 表示沒有變化
data['Li_count'] = data['Li_count'].fillna(0)

print("\n已添加各區里的權重統計量特徵。")

# 17. 定義特徵矩陣 X 和目標變量 y
feature_columns = ['權重', '顏色_數值', 'Size', 'Age', 'Floor'] + ['tag_' + tag.replace(' ', '_') for tag in selected_tags]

# 添加新的 'Li' 特徵
feature_columns += ['Li_權重_mean', 'Li_權重_median', 'Li_權重_std', 'Li_count']

# 18. 特徵標準化
scaler = StandardScaler()

# 標準化數值型特徵
data[numeric_cols := ['顏色_數值', 'Size', 'Age', 'Floor', 'Li_權重_mean', 'Li_權重_median', 'Li_權重_std', 'Li_count']] = \
    scaler.fit_transform(data[numeric_cols := ['顏色_數值', 'Size', 'Age', 'Floor', 'Li_權重_mean', 'Li_權重_median', 'Li_權重_std', 'Li_count']])

print("已對數值型特徵進行標準化。")

# 19. 創建特徵交互（增加權重影響力）
data['Size_Weight'] = data['Size'] * data['權重']
feature_columns.append('Size_Weight')
print("已創建 'Size_Weight' 特徵。")

# 20. 刪除包含 NaN 的行
initial_row_count = data.shape[0]
data_clean = data.dropna(subset=feature_columns + ['Price'])
cleaned_row_count = data_clean.shape[0]
rows_dropped = initial_row_count - cleaned_row_count
print(f"\n已刪除 {rows_dropped} 筆包含 NaN 的資料。")

# 21. 定義特徵矩陣 X 和目標變量 y（在清理後的資料上）
X = data_clean[feature_columns]
y = data_clean['Price']

# 確認特徵矩陣 X 是否包含 NaN
if X.isnull().values.any():
    print("特徵矩陣 X 包含 NaN 值。請檢查資料處理步驟。")
    sys.exit(1)
else:
    print("特徵矩陣 X 不包含 NaN 值。")

# 22. 定義交叉驗證
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 23. 定義 RandomForestModel 類，用於超參數調整和模型分析
class RandomForestModel:
    def __init__(self, **kwargs):
        # 預設超參數
        self.params = {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }

        # 更新預設參數
        self.params.update(kwargs)
        self.model = RandomForestRegressor(**self.params)
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def add_param(self, key, value):
        self.params[key] = value
        self.model.set_params(**{key: value})
        print(f"已添加參數 '{key}'，值為: {value}")
        
    def delete_param(self, key):
        if key in self.params:
            del self.params[key]
            # 重新實例化模型
            self.model = RandomForestRegressor(**self.params)
            print(f"已刪除參數 '{key}'。")
        else:
            print(f"參數 '{key}' 未找到。")
        
    def modify_param(self, key, value):
        if key in self.params:
            self.params[key] = value
            self.model.set_params(**{key: value})
            print(f"已修改參數 '{key}'，新值為: {value}")
        else:
            print(f"參數 '{key}' 未找到。")
        
    def display_params(self):
        print("當前超參數設定:")
        for key, value in self.params.items():
            print(f"{key}: {value}")
        
    def feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance}")
        
    def plot_feature_importance(self, feature_names):
        # 繪製特徵重要性圖表
        importances = pd.Series(self.model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.show()
        
    def residual_analysis(self, y_true, y_pred):
        # 計算殘差
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residual Analysis')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()
        
    def hyperparameter_tuning(self, X, y):
        # 定義超參數網格
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        # 使用 GridSearchCV 進行超參數調整
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        # 更新模型為最佳參數
        self.model = grid_search.best_estimator_
        self.params = grid_search.best_params_
        print("最佳參數：", self.params)
        
    def cross_validate_model(self, X, y, cv):
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        rmse_scores = np.sqrt(-scores)
        print(f"交叉驗證 RMSE: {rmse_scores.mean()} ± {rmse_scores.std()}")
        return rmse_scores

# 24. 實例化並訓練模型
rf_model = RandomForestModel()

# 顯示當前超參數
rf_model.display_params()

# 進行交叉驗證
print("\n進行初始模型的交叉驗證...")
rf_model.cross_validate_model(X, y, cv)

# 訓練模型
rf_model.train(X, y)

# 25. 使用交叉驗證預測並評估模型
predictions_log = cross_val_predict(rf_model.model, X, y, cv=cv, n_jobs=-1)
predictions = np.expm1(predictions_log)
y_exp = np.expm1(y)

# 評估模型性能
mse = mean_squared_error(y_exp, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_exp, predictions)

print(f"\n交叉驗證均方誤差 (MSE): {mse}")
print(f"交叉驗證均方根誤差 (RMSE): {rmse}")
print(f"交叉驗證 R^2 分數: {r2}")

# 繪製預測值與實際值的散佈圖
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_exp, y=predictions)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.plot([y_exp.min(), y_exp.max()], [y_exp.min(), y_exp.max()], 'r--')
plt.show()

# 繪製預測價格和實際價格的曲線圖
# 將數據按實際價格排序
sorted_indices = np.argsort(y_exp)
sorted_actual_prices = y_exp.values[sorted_indices]
sorted_predicted_prices = predictions[sorted_indices]

plt.figure(figsize=(12, 6))
plt.plot(sorted_actual_prices, label='Actual Price', color='blue')
plt.plot(sorted_predicted_prices, label='Predicted Price', color='orange')
plt.xlabel('Sample Index (sorted by Actual Price)')
plt.ylabel('Price')
plt.title('Actual Price vs Predicted Price Comparison')
plt.legend()
plt.show()

# 26. 模型分析
# 顯示特徵重要性
print("\n特徵重要性:")
rf_model.feature_importance(X.columns)

# 繪製特徵重要性圖表
rf_model.plot_feature_importance(feature_columns)

# 殘差分析
rf_model.residual_analysis(y, predictions_log)

# 27. 使用超參數調整提高模型性能
print("\n正在進行超參數調整，可能需要一些時間...")

rf_model.hyperparameter_tuning(X, y)

# 使用最佳參數的模型進行交叉驗證預測
predictions_log = cross_val_predict(rf_model.model, X, y, cv=cv, n_jobs=-1)
predictions = np.expm1(predictions_log)

# 重新評估模型性能
mse = mean_squared_error(y_exp, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_exp, predictions)

print(f"\n調整後的交叉驗證均方誤差 (MSE): {mse}")
print(f"調整後的交叉驗證均方根誤差 (RMSE): {rmse}")
print(f"調整後的交叉驗證 R^2 分數: {r2}")

# 繪製調整後模型的預測值與實際值的散佈圖
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_exp, y=predictions)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price (After Hyperparameter Tuning)')
plt.plot([y_exp.min(), y_exp.max()], [y_exp.min(), y_exp.max()], 'r--')
plt.show()

# 繪製調整後預測價格和實際價格的曲線圖
# 將數據按實際價格排序
sorted_indices = np.argsort(y_exp)
sorted_actual_prices = y_exp.values[sorted_indices]
sorted_predicted_prices = predictions[sorted_indices]

plt.figure(figsize=(12, 6))
plt.plot(sorted_actual_prices, label='Actual Price', color='blue')
plt.plot(sorted_predicted_prices, label='Predicted Price', color='orange')
plt.xlabel('Sample Index (sorted by Actual Price)')
plt.ylabel('Price')
plt.title('Actual vs Predicted Price Comparison (After Hyperparameter Tuning)')
plt.legend()
plt.show()

# 重新進行殘差分析
rf_model.residual_analysis(y, predictions_log)

# 28. 嘗試使用 XGBoost 模型
# ...（保持原代碼，省略部分）

# 29. 定義主函數以供外部調用（如 ASP.NET 網站）
def main():
    import sys
    import json

    # 需要的特徵列表
    # 現在包括 '縣市' 和 '區' 以便獲取 '權重' 和 '顏色_數值'
    input_feature_list = ['縣市', '區', 'Size', 'Age', 'Floor'] + ['tag_' + tag.replace(' ', '_') for tag in selected_tags]
    expected_args = len(input_feature_list) + 1  # 加上腳本名稱

    if len(sys.argv) != expected_args:
        print(f"用法: python {sys.argv[0]} 縣市 區 Size Age Floor " + " ".join(['tag_' + tag.replace(' ', '_') for tag in selected_tags]))
        sys.exit(1)

    # 從命令行獲取特徵值
    input_values = {}
    for i, feature in enumerate(input_feature_list, start=1):
        if 'tag_' in feature:
            try:
                input_values[feature] = int(sys.argv[i])
                if input_values[feature] not in [0, 1]:
                    raise ValueError
            except ValueError:
                print(f"錯誤: 特徵 '{feature}' 必須是整數（0 或 1）。")
                sys.exit(1)
        elif feature in ['Size', 'Age', 'Floor']:
            try:
                input_values[feature] = float(sys.argv[i])
            except ValueError:
                print(f"錯誤: 特徵 '{feature}' 必須是浮點數。")
                sys.exit(1)
        else:
            input_values[feature] = sys.argv[i]

    # 初始化輸入特徵字典
    input_features = dict.fromkeys(feature_columns, 0)

    # 提取 '縣市' 和 '區' 來獲取 '權重' 和 '顏色_數值'
    input_city = input_values['縣市']
    input_district = input_values['區']

    # 標準化 '縣市' 名稱：將 '臺' 替換為 '台' 並移除空格
    input_city = input_city.replace('臺', '台').replace(' ', '')
    input_district = input_district.replace(' ', '')  # 移除 '區' 欄位中的空格

    # 獲取對應的權重和顏色數值
    weight_record = weight_data[(weight_data['縣市'] == input_city) & (weight_data['區'] == input_district)]
    if not weight_record.empty:
        input_weight = weight_record['權重'].values[0]
        input_color = color_mapping.get(weight_record['顏色'].values[0], 0)
    else:
        # 如果找不到對應的區域，提示錯誤
        print(f"錯誤: 找不到縣市 '{input_city}' 和區 '{input_district}' 的權重和顏色資料。")
        sys.exit(1)

    # 更新輸入特徵字典
    input_features['權重'] = input_weight
    input_features['顏色_數值'] = input_color

    # 更新其他特徵
    for key, value in input_values.items():
        if key not in ['縣市', '區']:
            input_features[key] = value

    # 添加 'Li' 特徵
    # 獲取對應的區的里權重統計量
    li_stats_record = li_stats[(li_stats['縣市'] == input_city) & (li_stats['區'] == input_district)]
    if not li_stats_record.empty:
        input_features['Li_權重_mean'] = li_stats_record['Li_權重_mean'].values[0]
        input_features['Li_權重_median'] = li_stats_record['Li_權重_median'].values[0]
        input_features['Li_權重_std'] = li_stats_record['Li_權重_std'].values[0]
        input_features['Li_count'] = li_stats_record['Li_count'].values[0]
    else:
        # 如果找不到對應的區域，使用中位數或0
        input_features['Li_權重_mean'] = data['Li_權重_mean'].median()
        input_features['Li_權重_median'] = data['Li_權重_median'].median()
        input_features['Li_權重_std'] = 0
        input_features['Li_count'] = 0

    # 對數值型特徵進行標準化
    numeric_features = ['顏色_數值', 'Size', 'Age', 'Floor', 'Li_權重_mean', 'Li_權重_median', 'Li_權重_std', 'Li_count']
    input_features_numeric = np.array([[input_features[feature] for feature in numeric_features]])
    input_features_scaled = scaler.transform(input_features_numeric)[0]
    for i, feature in enumerate(numeric_features):
        input_features[feature] = input_features_scaled[i]

    # 創建特徵交互 'Size_Weight'
    input_features['Size_Weight'] = input_features['Size'] * input_features['權重']

    # 創建 DataFrame
    input_df = pd.DataFrame([input_features])

    # 確保欄位的順序正確
    input_df = input_df[feature_columns]

    # 進行預測（使用調整後的隨機森林模型）
    prediction_log = rf_model.predict(input_df)[0]
    prediction = np.expm1(prediction_log)

    # 輸出預測結果為 JSON 格式
    output = {"prediction": prediction}
    print(json.dumps(output, ensure_ascii=False))

if __name__ == "__main__":
    main()
