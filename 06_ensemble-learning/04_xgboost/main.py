import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
import traceback

# 读取数据
test_data = pd.read_csv("./datasets/红酒品质分类_test.csv")
train_data = pd.read_csv("./datasets/红酒品质分类_train.csv")
X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

# 自建模型（限制 XGBoost 内部线程，避免与 GridSearchCV 并行冲突）
estimator = xgb.XGBClassifier(
    objective='multi:softmax',   # 多分类
    num_class=6,                 # 红酒品质有 6 个等级
    random_state=66,
    n_jobs=1,
    verbosity=1,
)

## 随机搜索 + 分层 K 折（比 GridSearchCV 更省时）
param_dist = {
    "max_depth": [1, 2, 3, 4, 5],
    "n_estimators": [100, 110, 120, 130, 140],
    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=66)
# n_iter 设小以减少训练次数；n_jobs=-1 在 Grid/RandomSearch 层并行
rs = RandomizedSearchCV(estimator, param_dist, n_iter=20, cv=skf, n_jobs=-1, random_state=66)

# 计算类别均衡权重
sample_weights = class_weight.compute_sample_weight('balanced', y_train)

try:
    rs.fit(X_train, y_train, sample_weight=sample_weights)
except KeyboardInterrupt:
    print("训练被中断 (KeyboardInterrupt)。将使用当前已找到的最优模型（若有）。")
except Exception as e:
    print("运行时出错：", e)
    traceback.print_exc()

if hasattr(rs, "best_params_"):
    print(f"最优模型为：{rs.best_params_}")
    print(f"最优CV评分为:{rs.best_score_}")
    y_pred = rs.predict(X_test)
    print(classification_report(y_test, y_pred))
else:
    print("未找到有效的最优模型，请检查训练日志或缩小搜索空间以便调试。")