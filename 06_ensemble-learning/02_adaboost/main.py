"""
【知识点】集成学习 —— 自适应提升 (AdaBoost)
--------------------------------------------------------------------------------
1. Boosting 思想：串行训练多个弱分类器，放大前一轮被错分样本的权重。
2. 最终决策：将所有弱分类器按照各自准确率进行加权线性投票。
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def adaboost_classification():
    # ==================== 1. 数据加载与二分类过滤 ====================
    df = pd.read_csv("./datasets/wine0501.csv")

    # 过滤掉类别 1，仅保留类别 2 和 3 进行二分类演示
    df = df[df["Class label"] != 1]
    X = df[["Hue", "Alcohol"]]
    y = LabelEncoder().fit_transform(df["Class label"]) # 转换为 0/1 标签

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=66, stratify=y
    )

    # ==================== 2. AdaBoost 集成建模 ====================
    # 指定浅层决策树 (max_depth=3) 作为弱基学习器
    base_estimator = DecisionTreeClassifier(max_depth=3, random_state=66)
    
    estimator = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100,
        learning_rate=0.1,
        random_state=66
    )
    estimator.fit(X_train, y_train)

    # ==================== 3. 预测与评估 ====================
    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=== AdaBoost 弱分类器提升结果 ===")
    print(f"基分类器: 决策树 (max_depth=3) x 100 棵")
    print(f"测试集准确率: {acc:.2%}")

if __name__ == "__main__":
    adaboost_classification()