"""
💡【知识点】集成学习 —— 自适应提升算法 (AdaBoost)
--------------------------------------------------------------------------------
📌【核心思想与本质】
  1. Boosting 串行迭代：
     - 每一轮迭代中，被前一轮弱分类器错分的样本权重被放大，正确分类的样本权重被缩小。
     - 后续弱分类器重点攻坚前序分错的“硬骨头”样本。
     - 最终将所有弱分类器根据分类准确度进行加权线性组合。
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    try:
        df = pd.read_csv("./datasets/wine0501.csv")
    except Exception:
        print("请确保 ./datasets/wine0501.csv 存在。")
        return

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 弱基学习器：最大深度为 2 的决策树桩
    base_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    ada = AdaBoostClassifier(estimator=base_tree, n_estimators=50, learning_rate=0.5, random_state=42)
    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_test)
    print(f"=== AdaBoost (50 棵弱决策树桩) 测试准确率: {accuracy_score(y_test, y_pred):.2%} ===")

if __name__ == "__main__":
    main()
