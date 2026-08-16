"""
💡【知识点】决策树 (Decision Tree) 分类与泰坦尼克号生还预测
--------------------------------------------------------------------------------
📌【核心思想与本质】
  1. 基尼不纯度 (Gini)：递归选择最能降低不确定性的特征进行二叉分支划分。
  2. 预剪枝 (max_depth)：限制树的最大生长深度，防止决策树对训练集噪声过拟合。
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    try:
        df = pd.read_csv("./datasets/train.csv")
    except Exception:
        print("请确保 ./datasets/train.csv 数据集存在。")
        return

    # 1. 特征选取与缺失值填补
    X = df[["Pclass", "Sex", "Age"]].copy()
    X["Age"] = X["Age"].fillna(X["Age"].mean())
    X = pd.get_dummies(X, drop_first=True)
    y = df["Survived"]

    # 2. 数据切分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 决策树建模 (限制最大深度为 5 进行预剪枝)
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_clf.fit(X_train, y_train)

    y_pred = tree_clf.predict(X_test)
    print("=== 泰坦尼克号生还预测 (决策树 max_depth=5) ===")
    print(classification_report(y_test, y_pred, target_names=["遇难 (0)", "生还 (1)"]))

if __name__ == "__main__":
    main()
