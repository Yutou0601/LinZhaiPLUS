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

# 為每個標籤創建一個新的二元特徵欄位
for tag in unique_tags:
    col_name = 'tag_' + tag
    data[col_name] = data['Tags'].apply(lambda x: int(tag in x) if isinstance(x, str) else 0)

# 刪除不需要的欄位
data = data.drop(['ID', 'Name', 'Location', 'Tags'], axis=1)

# 處理數值型特徵
# 確保 'Floor' 和 'Age' 欄位為數值型
data['Floor'] = pd.to_numeric(data['Floor'], errors='coerce').fillna(0).astype(int)
data['Age'] = pd.to_numeric(data['Age'], errors='coerce').fillna(0).astype(float)

# 處理缺失值
data = data.fillna(0)

# 2. 定義特徵矩陣 X 和目標變量 y
X = data.drop('Price', axis=1)
y = data['Price']

# 保存特徵名稱以供後續使用
feature_names = X.columns.tolist()

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
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
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
predictions = rf_model.predict(X_test)

# 評估模型性能
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"均方誤差 (MSE): {mse}")
print(f"均方根誤差 (RMSE): {rmse}")
print(f"R^2 分數: {r2}")

# 輸出預測的價格和實際的價格
results = pd.DataFrame({'實際價格': y_test.reset_index(drop=True), '預測價格': predictions})
print("\n預測結果比較：")
print(results.head())

# 繪製預測價格與實際價格的比較圖
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test.reset_index(drop=True), label='Actual Price', color='blue')
plt.plot(range(len(predictions)), predictions, label='Predicted Price', color='red')
plt.legend()
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.show()

# 7. 模型分析
# 顯示特徵重要性
print("\n特徵重要性:")
rf_model.feature_importance(X.columns)

# 繪製特徵重要性圖表
rf_model.plot_feature_importance(feature_names)

# 殘差分析
rf_model.residual_analysis(X_test, y_test.reset_index(drop=True), predictions)

# 8. 使用超參數調整提高模型性能
print("\n正在進行超參數調整，可能需要一些時間...")

rf_model.hyperparameter_tuning(X_train, y_train)

# 使用最佳參數的模型進行預測
predictions = rf_model.predict(X_test)

# 重新評估模型性能
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"\n調整後的均方誤差 (MSE): {mse}")
print(f"調整後的均方根誤差 (RMSE): {rmse}")
print(f"調整後的 R^2 分數: {r2}")

# 重新進行殘差分析
rf_model.residual_analysis(X_test, y_test.reset_index(drop=True), predictions)

# 9. 定義主函數以供外部調用（如 ASP.NET 網站）
def main():
    import sys
    import json
    
    # 需要的特徵列表
    input_feature_list = ['Size', 'Age', 'Floor'] + ['tag_' + tag for tag in unique_tags]
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
    
    # 創建 DataFrame
    input_df = pd.DataFrame([input_features])
    
    # 確保欄位的順序正確
    input_df = input_df[feature_names]
    
    # 進行預測
    prediction = rf_model.predict(input_df)[0]
    
    # 輸出預測結果為 JSON 格式
    output = {"prediction": prediction}
    print(json.dumps(output, ensure_ascii=False))

if __name__ == "__main__":
    main()
