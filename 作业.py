import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('US-pumpkins.csv')

# 查看数据基本信息
print("数据集基本信息：")
print(df.info())
print("\n数据集前5行：")
print(df.head())

# 统计缺失值
print("\n各列缺失值数量：")
print(df.isnull().sum())

# 筛选有用的列
# 选择与城市、品种、日期、价格相关的列
useful_columns = ['City Name', 'Variety', 'Date', 'Low Price', 'High Price', 'Mostly Low', 'Mostly High', 'Origin', 'Item Size', 'Color']
df_selected = df[useful_columns].copy()

print("\n筛选后的数据集：")
print(df_selected.head())

# 处理缺失值
# 查看缺失值比例
missing_ratio = df_selected.isnull().sum() / len(df_selected)
print("\n各列缺失值比例：")
print(missing_ratio)

# 删除缺失值过多的列（缺失值比例>50%）
columns_to_drop = missing_ratio[missing_ratio > 0.5].index
df_processed = df_selected.drop(columns=columns_to_drop)

# 填充剩余缺失值
# 对于数值型列，用中位数填充
numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
df_processed[numeric_columns] = df_processed[numeric_columns].fillna(df_processed[numeric_columns].median())

# 对于分类型列，用众数填充
categorical_columns = df_processed.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

# 处理日期列
df_processed['Date'] = pd.to_datetime(df_processed['Date'], errors='coerce')
df_processed['Year'] = df_processed['Date'].dt.year
df_processed['Month'] = df_processed['Date'].dt.month
df_processed['Day'] = df_processed['Date'].dt.day
df_processed.drop('Date', axis=1, inplace=True)

# 编码分类型变量
label_encoders = {}
for col in categorical_columns:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

# 特征缩放
scaler = StandardScaler()
df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])

print("\n处理后的数据集：")
print(df_processed.head())

# 划分特征和目标变量
X = df_processed.drop(['Low Price', 'High Price', 'Mostly Low', 'Mostly High'], axis=1)
y_low = df_processed['Low Price']
y_high = df_processed['High Price']
y_mostly_low = df_processed['Mostly Low']
y_mostly_high = df_processed['Mostly High']

# 划分训练集和测试集
X_train, X_test, y_train_low, y_test_low = train_test_split(X, y_low, test_size=0.2, random_state=42)
X_train, X_test, y_train_high, y_test_high = train_test_split(X, y_high, test_size=0.2, random_state=42)
X_train, X_test, y_train_mostly_low, y_test_mostly_low = train_test_split(X, y_mostly_low, test_size=0.2, random_state=42)
X_train, X_test, y_train_mostly_high, y_test_mostly_high = train_test_split(X, y_mostly_high, test_size=0.2, random_state=42)

# 线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train_low)
y_pred_lr = lr_model.predict(X_test)

# 随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_low)
y_pred_rf = rf_model.predict(X_test)

# 评估模型
print("\n线性回归模型评估：")
print(f"均方误差: {mean_squared_error(y_test_low, y_pred_lr)}")
print(f"R²分数: {r2_score(y_test_low, y_pred_lr)}")

print("\n随机森林回归模型评估：")
print(f"均方误差: {mean_squared_error(y_test_low, y_pred_rf)}")
print(f"R²分数: {r2_score(y_test_low, y_pred_rf)}")

# 特征重要性分析（随机森林）
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('特征重要性')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.show()

# 恢复原始价格数据以便可视化
df_original = df_selected.copy()
df_original['Date'] = pd.to_datetime(df_original['Date'], errors='coerce')

# 按月份分组分析价格分布
df_original['Month'] = df_original['Date'].dt.month
monthly_prices = df_original.groupby('Month')[['Low Price', 'High Price']].mean()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Month', y='Low Price', data=df_original)
plt.title('不同月份的南瓜最低价格分布')
plt.xlabel('月份')
plt.ylabel('最低价格')
plt.show()

# 按品种分析平均价格
variety_prices = df_original.groupby('Variety')[['Low Price', 'High Price']].mean().sort_values('Low Price', ascending=False).head(10)

# 创建一个图形
plt.figure(figsize=(14, 8))
variety_prices.plot(kind='bar', color=['#FFA07A', '#20B2AA'], ax=plt.gca())

# 设置标题和标签
plt.title('不同品种的平均价格', fontsize=16)
plt.xlabel('品种', fontsize=12)
plt.ylabel('平均价格 (元)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(['最低价格', '最高价格'], fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
