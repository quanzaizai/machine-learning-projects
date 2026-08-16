"""
💡【知识点】集成学习 —— 梯度提升决策树 (GBDT)
--------------------------------------------------------------------------------
📌【核心思想与本质】
  1. 拟合残差：每一棵新生成的决策树不直接拟合标签 y，而是拟合前序所有树预测值累加后的“负梯度”（残差 Residuals）。
  2. 加法模型：不断纠正前序模型的预测误差，逐步逼近真实目标值。
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def main():
    try:
        df = pd.read_csv("./datasets/train.csv")
    except Exception:
        print("请确保 ./datasets/train.csv 存在。")
        return

    X = df[["Pclass", "Sex", "Age"]].copy()
    X["Age"] = X["Age"].fillna(X["Age"].mean())
    X = pd.get_dummies(X, drop_first=True)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbdt.fit(X_train, y_train)

    acc = gbdt.score(X_test, y_test)
    print(f"=== GBDT 梯度提升树测试准确率: {acc:.2%} ===")

if __name__ == "__main__":
    main()
