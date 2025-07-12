import pandas as pd
import matplotlib.pyplot as plt
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


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 处理缺失值
# 查看缺失值比例
missing_ratio = df_selected.isnull().sum() / len(df_selected)
print("\n各列缺失值比例：")
print(missing_ratio)

# 删除缺失值过多的列
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

# 改进的日期处理 - 添加季节特征
df_processed['Date'] = pd.to_datetime(df_processed['Date'], errors='coerce')
df_processed['Year'] = df_processed['Date'].dt.year
df_processed['Month'] = df_processed['Date'].dt.month
df_processed['Day'] = df_processed['Date'].dt.day
# 添加季节特征
df_processed['Season'] = df_processed['Month'] % 12 // 3 + 1
df_processed.drop('Date', axis=1, inplace=True)

# 改进分类变量编码
categorical_columns = df_processed.select_dtypes(include=['object']).columns
numeric_columns = df_processed.select_dtypes(include=[np.number]).columns

# 分离出目标变量，不对其进行缩放
target_columns = ['Low Price', 'High Price', 'Mostly Low', 'Mostly High']
numeric_columns = [col for col in numeric_columns if col not in target_columns]

# OneHot编码分类变量
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cols = encoder.fit_transform(df_processed[categorical_columns])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_columns))

# 合并数值型和编码后的分类变量
df_processed = pd.concat([
    df_processed[numeric_columns + target_columns].reset_index(drop=True),
    encoded_df.reset_index(drop=True)
], axis=1)

# 只对特征进行标准化，不对目标变量标准化
scaler = StandardScaler()
df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])

print("\n改进后的处理数据集：")
print(df_processed.head())