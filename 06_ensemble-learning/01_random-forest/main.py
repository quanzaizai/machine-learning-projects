"""
💡【知识点】集成学习 (Ensemble Learning) —— 随机森林 (Random Forest) 与网格调参
--------------------------------------------------------------------------------
📌【核心思想与本质】
  1. Bagging (Bootstrap Aggregating) 思想：
     - 样本扰动：对训练集有放回随机重采样。
     - 特征扰动：每个节点分裂时仅随机挑选子集特征。
     - 投票表决：并行构建多棵决策树，综合所有弱分类器的投票结果降低整体方差。
  2. GridSearchCV：自动化搜索最佳树棵数 (n_estimators) 与深度 (max_depth)。
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def random_forest_classification():
    # ==================== 1. 数据加载与特征工程 ====================
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
        X, y, test_size=0.2, random_state=66, stratify=y
    )

    # ==================== 2. 随机森林基线模型 ====================
    estimator = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=23)
    estimator.fit(X_train, y_train)
    baseline_acc = estimator.score(X_test, y_test)
    print(f"基线模型准确率 (树棵数=100, 深度=2): {baseline_acc:.2%}\n")

    # ==================== 3. 网格搜索自动化调优 ====================
    param_grid = {
        "max_depth": [2, 3, 4, 5, 6],
        "n_estimators": [100, 200, 300]
    }
    gs = GridSearchCV(estimator, param_grid, cv=3)
    gs.fit(X_train, y_train)

    print("=== 网格搜索优化结果 ===")
    print(f"  • 最优超参数组合: {gs.best_params_}")
    print(f"  • 调优后测试准确率: {gs.score(X_test, y_test):.2%}")

if __name__ == "__main__":
    random_forest_classification()
