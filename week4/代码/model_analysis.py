from IPython.display import display
import json

def model_analysis(results_df):
    print("\n模型评估结果 (按R2排序):")
    display(results_df[['Model', 'R2', 'CV R2', 'RMSE', 'MAE', 'Training Time']]
            .sort_values(by='R2', ascending=False))

    # 样本分析
    with open('sample_analysis.json', 'r') as f:
        samples = json.load(f)

    print("\n样本分析结果:")
    print("2个最正确预测:")
    for i, sample in enumerate([s for s in samples if s['type'] == 'correct'][:2]):
        print(f"\n样本 {i + 1}:")
        print(f"实际价格: {sample['actual']:.2f}, 预测价格: {sample['predicted']:.2f}")
        print(f"误差: {sample['error']:.2f}")
        print("关键特征:")
        print({k: v for k, v in sample['features'].items() if abs(v) > 0.5})

    print("\n2个最错误预测:")
    for i, sample in enumerate([s for s in samples if s['type'] == 'wrong'][:2]):
        print(f"\n样本 {i + 1}:")
        print(f"实际价格: {sample['actual']:.2f}, 预测价格: {sample['predicted']:.2f}")
        print(f"误差: {sample['error']:.2f}")
        print("关键特征:")
        print({k: v for k, v in sample['features'].items() if abs(v) > 0.5})

    # 显示特征相关性图
    print("\n特征可视化已保存为:")
    print("- processed_features_correlation.png (处理后特征相关性)")
    print("- feature_importances.png (特征重要性)")