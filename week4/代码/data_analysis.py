import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


def data_analysis(df):
    # 查看数据基本信息
    print("\n数据集基本信息：")
    print(df.info())
    print("\n数据集前5行：")
    print(df.head())

    # 删除无用列
    columns_to_drop = ['Grade', 'Unnamed: 24', 'Origin District', 'Sub Variety', 'Comments']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # 计算目标列
    df['mean_price'] = (df['Low Price'] + df['High Price']) / 2

    # 筛选有用的列
    useful_columns = ['City Name', 'Variety', 'Date', 'Low Price', 'High Price', 'mean_price',
                      'Mostly Low', 'Mostly High', 'Origin', 'Item Size', 'Color']
    df_selected = df[useful_columns].copy()

    # 处理缺失值
    df_selected = df_selected.dropna(subset=['mean_price'])

    # 删除重复值
    df_selected = df_selected.drop_duplicates()

    print("\n处理后的数据集前5行：")
    print(df_selected.head())

    # 特征重要性分析
    X_temp = pd.get_dummies(df_selected.drop(columns=['mean_price', 'Date']))
    y_temp = df_selected['mean_price']

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_temp, y_temp)

    # 交叉验证评估特征选择
    cv_scores = cross_val_score(rf, X_temp, y_temp, cv=5, scoring='r2')
    print(f"\n交叉验证R2分数(均值±标准差): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return df_selected