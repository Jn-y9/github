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
