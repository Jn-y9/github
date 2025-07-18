from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

def modeling(X, y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # 模型列表及配置
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    # 结果存储
    results = []
    json_results = {}
    sample_analysis = []

    for name, model in models.items():
        start = time.time()

        # 交叉验证
        cv_pred = cross_val_predict(model, X_train, y_train, cv=5)
        cv_r2 = r2_score(y_train, cv_pred)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        training_time = time.time() - start

        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CV R2': cv_r2,
            'Training Time': training_time
        })

        # 存储LGBM和XGBoost结果
        if name in ['LightGBM', 'XGBoost']:
            json_results[name] = {
                'RMSE': round(rmse, 3),
                'MAE': round(mae, 3),
                'R2': round(r2, 3),
                'CV R2': round(cv_r2, 3),
                'Training Time': round(training_time, 3)
            }

        # 样本分析 (使用表现最好的模型)
        if name == 'LightGBM':
            residuals = y_test - y_pred
            top_correct = np.argsort(np.abs(residuals))[:2]  # 最正确的2个样本
            top_wrong = np.argsort(np.abs(residuals))[-2:]  # 最错误的2个样本

            for idx in top_correct:
                sample_analysis.append({
                    'type': 'correct',
                    'actual': y_test.iloc[idx],
                    'predicted': y_pred[idx],
                    'error': residuals.iloc[idx],
                    'features': X_test.iloc[idx].to_dict()
                })

            for idx in top_wrong:
                sample_analysis.append({
                    'type': 'wrong',
                    'actual': y_test.iloc[idx],
                    'predicted': y_pred[idx],
                    'error': residuals.iloc[idx],
                    'features': X_test.iloc[idx].to_dict()
                })

            # 特征重要性可视化
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:20]
                sns.barplot(x=feature_imp.values, y=feature_imp.index)
                plt.title('Top 20 Feature Importances')
                plt.tight_layout()
                plt.savefig('feature_importances.png')
                plt.close()

    # 保存样本分析结果
    with open('sample_analysis.json', 'w') as f:
        json.dump(sample_analysis, f, indent=4)

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    return results_df, X_train, X_test, y_train, y_test