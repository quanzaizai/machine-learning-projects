"""
=============================================================================
💡【知识点】集成学习工业王者 —— XGBoost (eXtreme Gradient Boosting)
=============================================================================

📌【1. XGBoost 相比传统 GBDT 的革命性升级】
  - **二阶泰勒展开 (Second-Order Taylor Expansion)** :
    - GBDT 在优化损失函数时仅使用一阶导数（梯度 $g_i$）。
    - XGBoost 引入二阶导数（Hessian 矩阵 $h_i$），对目标函数进行了二阶泰勒级数展开，优化方向和步长极度精准！
  - **显式结构正则化项 (Explicit Regularization)** :
    - 目标函数中直接加入树复杂度惩罚：
      $$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
      (其中 $T$ 为叶子节点总数，$w$ 为叶子节点权重得分，$\gamma$ 和 $\lambda$ 控制抗过拟合强度)。
=============================================================================
"""

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def main() -> None:
    """XGBoost 红酒品质多分类主程序"""
    print("=" * 52)
    print("      ⚡ 集成学习工业级霸主 —— XGBoost 多分类实战      ")
    print("=" * 52)

    if xgb is None:
        print("⚠️ 未检测到 xgboost 运行环境，可通过 `uv pip install xgboost` 安装体验。")
        return

    # 【步骤 1】加载数据
    try:
        df = pd.read_csv("./datasets/wine0501.csv")
    except Exception:
        print("⚠️ 请确保 ./datasets/wine0501.csv 存在。")
        return

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # XGBoost 严格要求多分类标签必须是从 0 开始的非负连续整数 (0, 1, 2...)
    y = y - y.min()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 【步骤 2】配置并拟合 XGBoost 分类器
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        random_state=42
    )
    model.fit(X_train, y_train)

    # 【步骤 3】预测与多分类报告评估
    y_pred = model.predict(X_test)

    print("\n" + "=" * 52)
    print("📋 XGBoost 多分类精准评估报告:")
    print("-" * 52)
    print(classification_report(y_test, y_pred))
    print("=" * 52)


if __name__ == "__main__":
    main()
