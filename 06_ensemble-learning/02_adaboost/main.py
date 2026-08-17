"""
=============================================================================
💡【知识点】集成学习 (Ensemble Learning) —— 自适应提升算法 (AdaBoost)
=============================================================================

📌【1. Boosting 串行提升机制】
  - 核心哲学：知错就改，重点攻坚。
  - **样本权重动态自适应调整** :
    - 初始时所有样本权重相等 ($w_i = 1/N$)。
    - 每一轮训练出一个弱学习器（如极浅的决策树桩 Decision Stump）。
    - 检查预测结果：**上一轮被分错的样本，在下一轮中的权重被成倍放大**；被正确分类的样本权重被缩小。
    - 下一个弱分类器被迫全力聚焦在那些“极其顽固的硬骨头样本”上！
  - **最终投票加权求和** :
    - 错误率越低的弱分类器，在最终投票时拥有越大的表决话语权系数 $\alpha_m$。
=============================================================================
"""

# ==================== 0. Boosting 自适应提升集成库引入 ====================
import pandas as pd                                  # 数据处理库：红酒理化指标数据加载
from sklearn.ensemble import AdaBoostClassifier      # 集成算法：AdaBoost 自适应提升分类器 (动态调整错分样本权重)
from sklearn.metrics import accuracy_score           # 评估库：计算最终加权投票的分类准确率
from sklearn.model_selection import train_test_split # 数据切分：划分训练与测试评估集
from sklearn.tree import DecisionTreeClassifier      # 弱基学习器：单层/双层浅决策树桩 (Decision Stump)


def main() -> None:
    """AdaBoost 红酒品质分类主程序"""
    print("=" * 52)
    print("      🚀 集成学习 Boosting 代表 —— AdaBoost 串行提升      ")
    print("=" * 52)

    # 【步骤 1】加载经典红酒分类数据集
    try:
        df = pd.read_csv("./datasets/wine0501.csv")
    except Exception:
        print("⚠️ 请确保 ./datasets/wine0501.csv 存在。")
        return

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print(f"📦 数据规模: {X.shape[0]} 例样本 | 理化特征维度: {X.shape[1]} 维")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 【步骤 2】配置基础弱学习器 (最大深度为 2 的决策树桩 Decision Stump)
    base_tree = DecisionTreeClassifier(max_depth=2, random_state=42)

    # 【步骤 3】构建 AdaBoost 集成模型 (50 棵弱决策树桩，学习率 0.5)
    ada = AdaBoostClassifier(
        estimator=base_tree, 
        n_estimators=50, 
        learning_rate=0.5, 
        random_state=42
    )
    ada.fit(X_train, y_train)

    # 【步骤 4】评估最终泛化能力
    y_pred = ada.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 52)
    print("🎯 AdaBoost 弱分类器加权组合评估:")
    print(f"  • 基学习器类型 : 深度为 2 的轻量决策树桩 (Decision Stump)")
    print(f"  • 弱分类器数量 : 50 棵")
    print(f"  • 最终测试准确率 : {test_acc:.2%}")
    print("=" * 52)


if __name__ == "__main__":
    main()
