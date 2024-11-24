# house_price_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 1. 加载并预处理数据集
# 请确保将 'House_Rent_Info.csv' 替换为您的实际数据集路径
data = pd.read_csv('House_Rent_Info.csv')

# 处理分类变量
categorical_cols = ['Name', 'City', 'Location', 'HouseType', 'Pattern', 'Tags', 'Environment']

# 初始化 LabelEncoder
le = LabelEncoder()

# 对每个分类列进行编码
for col in categorical_cols:
    data[col] = le.fit_transform(data[col].astype(str))

# 2. 定义特征矩阵 X 和目标变量 y
X = data.drop('Price', axis=1)
y = data['Price']

# 3. 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 开发者接口类，用于超参数调整和模型分析
class RandomForestModel:
    def __init__(self, **kwargs):
        # 默认超参数
        self.params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42
        }
        # 更新默认参数
        self.params.update(kwargs)
        self.model = RandomForestRegressor(**self.params)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def add_param(self, key, value):
        self.params[key] = value
        self.model.set_params(**{key: value})
        print(f"已添加参数 '{key}'，值为: {value}")
    
    def delete_param(self, key):
        if key in self.params:
            del self.params[key]
            # 重新实例化模型
            self.model = RandomForestRegressor(**self.params)
            print(f"已删除参数 '{key}'。")
        else:
            print(f"参数 '{key}' 未找到。")
    
    def modify_param(self, key, value):
        if key in self.params:
            self.params[key] = value
            self.model.set_params(**{key: value})
            print(f"已修改参数 '{key}'，新值为: {value}")
        else:
            print(f"参数 '{key}' 未找到。")
    
    def display_params(self):
        print("当前超参数设置:")
        for key, value in self.params.items():
            print(f"{key}: {value}")
    
    def feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance}")
    
    def visualize_tree(self, X_columns, tree_index=0, text_color='black'):
        # 可视化随机森林中的一棵树
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
        # 修改文字颜色
        for ann in annotations:
            ann.set_color(text_color)
        plt.show()

# 5. 实例化并训练模型
rf_model = RandomForestModel()

# 显示当前超参数
rf_model.display_params()

# 训练模型
rf_model.train(X_train, y_train)

# 6. 进行预测并评估模型
predictions = rf_model.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"均方误差 (MSE): {mse}")
print(f"R^2 分数: {r2}")

# 输出预测的价格和实际的价格
results = pd.DataFrame({'实际价格': y_test.reset_index(drop=True), '预测价格': predictions})
print("\n预测结果对比：")
print(results.head())

# 绘制预测价格与实际价格的对比图
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test.reset_index(drop=True), label='实际价格', color='blue')
plt.plot(range(len(predictions)), predictions, label='预测价格', color='red')
plt.legend()
plt.title('实际价格 vs 预测价格')
plt.xlabel('样本编号')
plt.ylabel('价格')
plt.show()

# 7. 模型分析
# 显示特征重要性
print("\n特征重要性:")
rf_model.feature_importance(X.columns)

# 可视化决策树（可选）
# 例如，将文字颜色设置为红色
# rf_model.visualize_tree(X.columns, tree_index=0, text_color='red')

# 8. 使用关键字调整超参数
# 添加新超参数
rf_model.add_param('bootstrap', False)
rf_model.display_params()

# 修改现有超参数
rf_model.modify_param('n_estimators', 200)
rf_model.display_params()

# 删除超参数
rf_model.delete_param('max_depth')
rf_model.display_params()

# 9. 使用更新的超参数重新训练模型
rf_model.train(X_train, y_train)

# 重新评估模型
predictions = rf_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n调整超参数后:")
print(f"均方误差 (MSE): {mse}")
print(f"R^2 分数: {r2}")

# 输出预测的价格和实际的价格
results = pd.DataFrame({'实际价格': y_test.reset_index(drop=True), '预测价格': predictions})
print("\n调整超参数后预测结果对比：")
print(results.head())

# 绘制预测价格与实际价格的对比图
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test.reset_index(drop=True), label='实际价格', color='blue')
plt.plot(range(len(predictions)), predictions, label='预测价格', color='red')
plt.legend()
plt.title('调整超参数后 实际价格 vs 预测价格')
plt.xlabel('样本编号')
plt.ylabel('价格')
plt.show()

# 10. 可视化更新后的决策树（可选）
# 例如，将文字颜色设置为红色
# rf_model.visualize_tree(X.columns, tree_index=0, text_color='red')
