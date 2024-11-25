# house_price_prediction.py

import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# 1. 載入並預處理資料集
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'House_Rent_Info.csv')
data = pd.read_csv(csv_path)

# 處理 'Tags' 欄位，提取關鍵字並創建二元特徵
# 提取所有唯一的標籤
unique_tags = set()

for tags_str in data['Tags'].dropna():
    tags_list = [tag.strip() for tag in tags_str.split(',')]
    unique_tags.update(tags_list)

# 建立中文標籤到英文標籤的對照表
tag_translation = {
    '公園附近': 'Near Park',
    '靠近學校': 'Near School',
    '面向陽光': 'Sun Facing',
    # 在此添加所有標籤的對應翻譯
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
threshold = 50  # 根據實際情況調整
selected_tags = [tag for tag, count in tag_counts.items() if count >= threshold]

# 為每個選定的標籤創建一個新的二元特徵欄位
for tag in selected_tags:
    col_name = 'tag_' + tag.replace(' ', '_')
    data[col_name] = data['Tags_English'].apply(lambda x: int(tag in x) if isinstance(x, str) else 0)

# 刪除不需要的欄位
data = data.drop(['ID', 'Name', 'Location', 'Tags', 'Tags_English'], axis=1)

# 處理數值型特徵
# 確保 'Floor' 和 'Age' 欄位為數值型
data['Floor'] = pd.to_numeric(data['Floor'], errors='coerce')
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

# 針對性地處理缺失值
data['Floor'] = data['Floor'].fillna(data['Floor'].median())
data['Age'] = data['Age'].fillna(data['Age'].median())

# 去除價格的異常值
# 假設合理的價格範圍為 50 萬到 1 億
data = data[(data['Price'] >= 500000) & (data['Price'] <= 100000000)]

# 對價格進行對數變換
data['Price'] = np.log1p(data['Price'])

# 2. 定義特徵矩陣 X 和目標變量 y
# 選擇需要的特徵
feature_columns = ['Size', 'Age', 'Floor'] + ['tag_' + tag.replace(' ', '_') for tag in selected_tags]
X = data[feature_columns]
y = data['Price']

# 保存特徵名稱以供後續使用
feature_names = X.columns.tolist()

# 特徵標準化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X[['Size', 'Age', 'Floor']] = scaler.fit_transform(X[['Size', 'Age', 'Floor']])

# 3. 將資料集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 開發者介面類，用於超參數調整和模型分析
class RandomForestModel:
    def __init__(self, **kwargs):
        # 預設超參數
        self.params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
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
        
    def visualize_tree(self, X_columns, tree_index=0, text_color='black'):
        # 可視化隨機森林中的一棵樹
        estimator = self.model.estimators_[tree_index]
        plt.figure(figsize=(20, 10))
        annotations = tree.plot_tree(
            estimator,
            feature_names=X_columns,
            filled=True,
            fontsize=10,
            impurity=False,
            proportion=True,
            rounded=True,
            node_ids=False,
            precision=2
        )
        # 修改文字顏色
        for ann in annotations:
            ann.set_color(text_color)
        plt.show()
        
    def residual_analysis(self, X_test, y_test, predictions):
        # 計算殘差
        residuals = y_test - predictions
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=predictions, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residual Analysis')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()
        
    def hyperparameter_tuning(self, X, y):
        # 定義超參數網格
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']  # 移除 'auto'
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

# 5. 實例化並訓練模型
rf_model = RandomForestModel()

# 顯示當前超參數
rf_model.display_params()

# 訓練模型
rf_model.train(X_train, y_train)

# 6. 進行預測並評估模型
predictions_log = rf_model.predict(X_test)
predictions = np.expm1(predictions_log)
y_test_exp = np.expm1(y_test)

# 評估模型性能
mse = mean_squared_error(y_test_exp, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_exp, predictions)

print(f"均方誤差 (MSE): {mse}")
print(f"均方根誤差 (RMSE): {rmse}")
print(f"R^2 分數: {r2}")

# 輸出預測的價格和實際的價格
results = pd.DataFrame({'實際價格': y_test_exp.reset_index(drop=True), '預測價格': predictions})
print("\n預測結果比較：")
print(results.head())

# 繪製預測值與實際值的散佈圖
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_exp, y=predictions)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.plot([y_test_exp.min(), y_test_exp.max()], [y_test_exp.min(), y_test_exp.max()], 'r--')
plt.show()

# 繪製預測價格和實際價格的曲線圖
# 將測試集按實際價格排序
sorted_indices = np.argsort(y_test_exp)
sorted_actual_prices = y_test_exp.values[sorted_indices]
sorted_predicted_prices = predictions[sorted_indices]

plt.figure(figsize=(12, 6))
plt.plot(sorted_actual_prices, label='Actual Price', color='blue')
plt.plot(sorted_predicted_prices, label='Predicted Price', color='orange')
plt.xlabel('Sample Index (sorted by Actual Price)')
plt.ylabel('Price')
plt.title('Actual Price vs Predicted Price Comparison')
plt.legend()
plt.show()

# 7. 模型分析
# 顯示特徵重要性
print("\n特徵重要性:")
rf_model.feature_importance(X.columns)

# 繪製特徵重要性圖表
rf_model.plot_feature_importance(feature_names)

# 殘差分析
rf_model.residual_analysis(X_test, y_test, predictions_log)

# 8. 使用超參數調整提高模型性能
print("\n正在進行超參數調整，可能需要一些時間...")

rf_model.hyperparameter_tuning(X_train, y_train)

# 使用最佳參數的模型進行預測
predictions_log = rf_model.predict(X_test)
predictions = np.expm1(predictions_log)

# 重新評估模型性能
mse = mean_squared_error(y_test_exp, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_exp, predictions)

print(f"\n調整後的均方誤差 (MSE): {mse}")
print(f"調整後的均方根誤差 (RMSE): {rmse}")
print(f"調整後的 R^2 分數: {r2}")

# 繪製調整後模型的預測值與實際值的散佈圖
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_exp, y=predictions)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price (After Hyperparameter Tuning)')
plt.plot([y_test_exp.min(), y_test_exp.max()], [y_test_exp.min(), y_test_exp.max()], 'r--')
plt.show()

# 繪製調整後預測價格和實際價格的曲線圖
# 將測試集按實際價格排序
sorted_indices = np.argsort(y_test_exp)
sorted_actual_prices = y_test_exp.values[sorted_indices]
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
rf_model.residual_analysis(X_test, y_test, predictions_log)

# 9. 定義主函數以供外部調用（如 ASP.NET 網站）
def main():
    import sys
    import json

    # 需要的特徵列表
    input_feature_list = ['Size', 'Age', 'Floor'] + ['tag_' + tag.replace(' ', '_') for tag in selected_tags]
    expected_args = len(input_feature_list) + 1  # 加上腳本名稱

    if len(sys.argv) != expected_args:
        print(f"用法: python {sys.argv[0]} " + " ".join(input_feature_list))
        sys.exit(1)

    # 從命令行獲取特徵值
    input_values = {}
    for i, feature in enumerate(input_feature_list, start=1):
        if 'tag_' in feature:
            input_values[feature] = int(sys.argv[i])
        else:
            input_values[feature] = float(sys.argv[i])

    # 初始化輸入特徵字典
    input_features = dict.fromkeys(feature_names, 0)

    # 更新輸入特徵字典
    for key, value in input_values.items():
        if key in input_features:
            input_features[key] = value

    # 對數值型特徵進行標準化
    numeric_features = ['Size', 'Age', 'Floor']
    input_features_numeric = np.array([[input_features[feature] for feature in numeric_features]])
    input_features_scaled = scaler.transform(input_features_numeric)[0]
    for i, feature in enumerate(numeric_features):
        input_features[feature] = input_features_scaled[i]

    # 創建 DataFrame
    input_df = pd.DataFrame([input_features])

    # 確保欄位的順序正確
    input_df = input_df[feature_names]

    # 進行預測
    prediction_log = rf_model.predict(input_df)[0]
    prediction = np.expm1(prediction_log)

    # 輸出預測結果為 JSON 格式
    output = {"prediction": prediction}
    print(json.dumps(output, ensure_ascii=False))

if __name__ == "__main__":
    main()
