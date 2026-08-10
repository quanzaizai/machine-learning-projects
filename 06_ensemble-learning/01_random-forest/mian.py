"""
💡【知识点】集成学习 (Ensemble Learning) —— 随机森林 (Random Forest) 与网格调参
--------------------------------------------------------------------------------
📌【概念与本质】
  1. Bagging (Bootstrap Aggregating) 核心思想：
     - 样本扰动：对训练集进行有放回随机采样 (Bootstrap)。
     - 特征扰动：在每个树节点分裂时随机选择子集特征。
     - 投票聚合：并行构建多棵决策树，综合所有弱分类器的投票结果降低整体方差。
  2. 超参数调优：利用 GridSearchCV 寻找最佳树棵数 (n_estimators) 与最大深度 (max_depth)。

📌【架构与模块分工】
  1. 数据清洗与独热编码 (fillna, get_dummies)
  2. 随机森林基线模型训练与评估
  3. GridSearchCV 自动化超参数调优与精度对比
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def random_forest_classification():
    # ==================== 1. 数据加载与特征工程 ====================
    df = pd.read_csv("./datasets/train.csv")

    X = df[["Pclass", "Sex", "Age"]].copy()
    X["Age"] = X["Age"].fillna(X["Age"].mean())
    X = pd.get_dummies(X)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=66, stratify=y
    )

    # ==================== 2. 随机森林基线模型 ====================
    estimator = RandomForestClassifier(
        n_estimators=100,
        max_depth=2,
        random_state=23
    )
    estimator.fit(X_train, y_train)
    baseline_acc = estimator.score(X_test, y_test)
    print("=== 随机森林基线模型 (max_depth=2, n_estimators=100) ===")
    print(f"基线测试准确率: {baseline_acc:.2%}\n")

    # ==================== 3. 网格搜索调参 ====================
    param_grid = {
        "max_depth": [1, 2, 3, 4, 5, 6],
        "n_estimators": [100, 200, 300, 400, 500]
    }
    gs = GridSearchCV(estimator, param_grid, cv=3)
    gs.fit(X_train, y_train)

    print("=== 网格搜索优化结果 ===")
    print(f"最优超参数组合: {gs.best_params_}")
    print(f"调优后测试准确率: {gs.score(X_test, y_test):.2%}")

if __name__ == "__main__":
    random_forest_classification()
