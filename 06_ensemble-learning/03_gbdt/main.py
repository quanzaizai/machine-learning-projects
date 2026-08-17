"""
=============================================================================
💡【知识点】集成学习 —— 梯度提升决策树 (Gradient Boosting Decision Tree / GBDT)
=============================================================================

📌【1. GBDT 的核心机制 —— 拟合负梯度 (残差)】
  - 核心思想：
    - AdaBoost 是通过改变样本权重来让后序弱分类器关注错题。
    - **GBDT 则是通过让每一棵新树去直接拟合前面所有树预测结果的“残差 / 负梯度 (Residuals)”**！
  - 加法模型 (Additive Model) :
    $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$
    - 第 1 棵树预测一个基础值，发现与真实值相差 $+10$（残差）；
    - 第 2 棵树专门去预测这 $+10$ 的误差，把残差缩小到 $+2$；
    - 第 3 棵树专门去预测这 $+2$ 的微小误差...
    - 最终累加所有树的预测，无限逼近真实目标值！
=============================================================================
"""

# ==================== 0. 梯度提升决策树 GBDT 模块引入 ====================
import pandas as pd                                  # 数据处理库：缺失值与离散变量预处理
from sklearn.ensemble import GradientBoostingClassifier # 集成算法：GBDT 梯度提升分类器 (迭代拟合前序预测负梯度残差)
from sklearn.model_selection import train_test_split # 数据划分：训练集与测试集划分


def main() -> None:
    """GBDT 梯度提升决策树主程序"""
    print("=" * 52)
    print("      📈 梯度提升树典范 —— GBDT (Gradient Boosting)      ")
    print("=" * 52)

    # 【步骤 1】加载数据并完成特征预处理
    try:
        df = pd.read_csv("./datasets/train.csv")
    except Exception:
        print("⚠️ 请确保 ./datasets/train.csv 存在。")
        return

    X = df[["Pclass", "Sex", "Age"]].copy()
    X["Age"] = X["Age"].fillna(X["Age"].mean())
    X = pd.get_dummies(X, drop_first=True)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 【步骤 2】构建并训练 GBDT 模型
    # n_estimators=100 (迭代 100 轮不断拟合前序残差)
    # learning_rate=0.1 (学习率收缩，防止步子过大步过拟合)
    # max_depth=3 (经典的弱回归树深度)
    gbdt = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    gbdt.fit(X_train, y_train)

    # 【步骤 3】评估测试集泛化得分
    acc = gbdt.score(X_test, y_test)

    print("\n" + "=" * 52)
    print("🎯 GBDT 残差提升模型评估:")
    print(f"  • 弱基回归树棵数 : 100 棵")
    print(f"  • 单树最大深度   : max_depth = 3")
    print(f"  • 最终测试准确率 : {acc:.2%}")
    print("=" * 52)


if __name__ == "__main__":
    main()
