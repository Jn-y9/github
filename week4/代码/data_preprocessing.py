import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def data_preprocessing(df):
    # 处理缺失值
    missing_ratio = df.isnull().sum() / len(df)
    print("\n各列缺失值比例：")
    print(missing_ratio)

    # 删除缺失值过多的列
    columns_to_drop = missing_ratio[missing_ratio > 0.3].index
    df_processed = df.drop(columns=columns_to_drop)

    # 填充剩余缺失值
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_columns] = df_processed[numeric_columns].fillna(df_processed[numeric_columns].median())

    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

    # 日期处理
    df_processed['Date'] = pd.to_datetime(df_processed['Date'], errors='coerce')
    df_processed['Year'] = df_processed['Date'].dt.year
    df_processed['Month'] = df_processed['Date'].dt.month
    df_processed['Day'] = df_processed['Date'].dt.day
    df_processed['DayOfWeek'] = df_processed['Date'].dt.dayofweek
    df_processed['Season'] = df_processed['Month'] % 12 // 3 + 1
    df_processed.drop('Date', axis=1, inplace=True)

    # 分类变量编码
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns

    # 分离出目标变量
    target_columns = ['Low Price', 'High Price', 'mean_price', 'Mostly Low', 'Mostly High']
    numeric_columns = [col for col in numeric_columns if col not in target_columns]

    # OneHot编码
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cols = encoder.fit_transform(df_processed[categorical_columns])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_columns))

    # 合并数据
    df_processed = pd.concat([
        df_processed[numeric_columns + target_columns].reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ], axis=1)

    # 数据归一化
    scaler = StandardScaler()
    df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])

    # 预处理后特征相关性热力图
    plt.figure(figsize=(16, 12))
    corr_matrix = df_processed[numeric_columns + ['mean_price']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Processed Features Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('processed_features_correlation.png')
    plt.close()

    print("\n处理后的数据集前5行：")
    print(df_processed.head())

    # 准备特征和目标
    X = df_processed.drop(columns=target_columns)
    y = df_processed['mean_price']

    return X, y