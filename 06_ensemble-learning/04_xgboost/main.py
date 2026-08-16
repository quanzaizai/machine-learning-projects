"""
💡【知识点】极端梯度提升 (XGBoost) —— 工业级突破与调参
--------------------------------------------------------------------------------
📌【核心思想与本质】
  1. 二阶泰勒展开：相比传统 GBDT 仅使用一阶导数，XGBoost 使用损失函数的二阶导数信息，优化方向更精确。
  2. 显式正则化：目标函数中加入叶子节点个数和叶子权重 L2 范数惩罚项，抑制模型过拟合。
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
try:
    import xgboost as xgb
except ImportError:
    xgb = None

def main():
    if xgb is None:
        print("未安装 xgboost 库，可通过 `uv pip install xgboost` 安装体验。")
        return

    try:
        df = pd.read_csv("./datasets/wine0501.csv")
    except Exception:
        print("请确保 ./datasets/wine0501.csv 存在。")
        return

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # 将标签映射为 0 开始的连续整数
    y = y - y.min()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== XGBoost 多分类评估报告 ===")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
