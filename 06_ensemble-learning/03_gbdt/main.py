"""
【知识点】集成学习 —— 梯度提升决策树 (GBDT)
--------------------------------------------------------------------------------
1. 梯度提升：通过迭代训练新树来拟合前序所有树累加后的残差（负梯度）。
2. 参数优化：结合学习率 (learning_rate) 与树棵数 (n_estimators) 平衡拟合与泛化。
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def gbdt_classification():
    # ==================== 1. 数据加载与预处理 ====================
    df = pd.read_csv("./datasets/train.csv")

    X = df[["Pclass", "Sex", "Age"]].copy()
    X["Age"] = X["Age"].fillna(X["Age"].mean())
    X = pd.get_dummies(X)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=66, stratify=y
    )

    # ==================== 2. GBDT 基线模型 ====================
    estimator = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=66
    )
    estimator.fit(X_train, y_train)
    print("=== GBDT 梯度提升决策树基线模型 ===")
    print(f"基线测试准确率: {estimator.score(X_test, y_test):.2%}\n")

    # ==================== 3. 学习率与树数量网格搜索 ====================
    param_grid = {
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [100, 200, 300]
    }
    gs = GridSearchCV(estimator, param_grid, cv=3)
    gs.fit(X_train, y_train)

    print("=== GBDT 网格搜索优化结果 ===")
    print(f"最优超参数组合: {gs.best_params_}")
    print(f"调优后测试准确率: {gs.score(X_test, y_test):.2%}")

if __name__ == "__main__":
    gbdt_classification()