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

# 删除缺失值过多的列（缺失值比例>30%）
columns_to_drop = missing_ratio[missing_ratio > 0.3].index
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
