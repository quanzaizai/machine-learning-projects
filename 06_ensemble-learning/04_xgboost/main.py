"""
【知识点】XGBoost 工业级极端梯度提升
--------------------------------------------------------------------------------
1. 核心改进：目标函数采用二阶泰勒展开 + 显式加入正则化项惩罚树复杂度。
2. 调参加速：RandomizedSearchCV 结合分层 K 折交叉验证高效搜索超参数。
--------------------------------------------------------------------------------
"""

import traceback
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils import class_weight
import xgboost as xgb

def xgboost_multiclass_classification():
    # ==================== 1. 数据加载 ====================
    train_data = pd.read_csv("./datasets/红酒品质分类_train.csv")
    test_data = pd.read_csv("./datasets/红酒品质分类_test.csv")

    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    # ==================== 2. XGBoost 多分类基模型 ====================
    estimator = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=6, # 6 个红酒品质等级
        random_state=66,
        n_jobs=1,
        verbosity=0
    )

    # ==================== 3. 随机搜索超参数空间 ====================
    param_dist = {
        "max_depth": [3, 4, 5, 6],
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.05, 0.1, 0.2]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=66)
    rs = RandomizedSearchCV(
        estimator,
        param_dist,
        n_iter=10,
        cv=skf,
        n_jobs=-1,
        random_state=66
    )

    # 计算样本均衡权重
    sample_weights = class_weight.compute_sample_weight("balanced", y_train)

    # ==================== 4. 模型训练与评估 ====================
    try:
        rs.fit(X_train, y_train, sample_weight=sample_weights)
        print("=== XGBoost 随机搜索与 5 折交叉验证优化结果 ===")
        print(f"最优参数组合: {rs.best_params_}")
        print(f"最优 CV 得分: {rs.best_score_:.2%}\n")

        y_pred = rs.predict(X_test)
        print("=== 独立测试集详细分类报告 ===")
        print(classification_report(y_test, y_pred))
    except Exception as e:
        print(f"【运行异常】: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    xgboost_multiclass_classification()