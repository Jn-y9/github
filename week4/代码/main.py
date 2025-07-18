import pandas as pd
import warnings
from data_analysis import data_analysis
from data_preprocessing import data_preprocessing
from modeling import modeling
from model_analysis import model_analysis


def main():
    # 忽略警告
    warnings.filterwarnings('ignore')

    # 读取数据
    df = pd.read_csv('US-pumpkins.csv')

    # 数据分析阶段 (包含特征选择和交叉验证)
    df_selected = data_analysis(df)

    # 数据处理阶段
    X, y = data_preprocessing(df_selected)

    # 建模阶段 (包含样本分析)
    results_df, X_train, X_test, y_train, y_test = modeling(X, y)

    # 模型分析阶段 (展示结果)
    model_analysis(results_df)


if __name__ == "__main__":
    main()