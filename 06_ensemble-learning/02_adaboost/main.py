"""
💡【知识点】集成学习 (Ensemble Learning) —— 自适应提升 (AdaBoost)
--------------------------------------------------------------------------------
📌【概念与本质】
  1. Boosting 核心思想：串行迭代训练多个弱分类器（Base Learners）。
  2. 样本权重自适应机制：
     - 每一轮迭代中，前一轮被错误分类的样本权重被放大，正确分类的样本权重被缩小。
     - 后续弱分类器会重点关注难以分类的“硬骨头”样本。
     - 最终各弱分类器按照自身分类准确率加权线性组合输出。

📌【架构与模块分工】
  1. 红酒数据集探索与多分类转二分类清洗 (LabelEncoder)
  2. 构造弱决策树基分类器 (DecisionTreeClassifier, max_depth=3)
  3. AdaBoost 集成拟合与分类评估
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