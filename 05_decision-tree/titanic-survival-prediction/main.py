"""
=============================================================================
💡【知识点】树模型经典实战 —— 决策树与泰坦尼克号生还预测 (Titanic Survival)
=============================================================================

📌【1. 决策树 (Decision Tree) 的划分逻辑】
  - 核心思想：通过一系列类似于 `if-else` 的二叉规则对特征空间进行正交切分。
  - 基尼不纯度准则 (Gini Impurity) :
    $$\text{Gini}(D) = 1 - \sum_{k=1}^{|y|} p_k^2$$
    - 基尼值越小，纯度越高，不确定性越低。
    - 算法在每个分裂节点遍历所有特征的所有可能切分点，选取**基尼增益最大**的特征进行分支。

📌【2. 预剪枝 (Pre-pruning) 防过拟合关键】
  - 树如果不加限制地任意生长，最终叶子节点可能只包含 1 个样本，导致在训练集上 100% 准确，在测试集上严重过拟合。
  - `max_depth=5`（限制树的最大深度）是一种经典的预剪枝手段，可大幅增强模型的泛化能力。
=============================================================================
"""

# ==================== 0. 树模型算法与特征处理库引入 ====================
import pandas as pd                                  # 数据清洗库：缺失值填充 (fillna) 与哑变量转换
from sklearn.metrics import classification_report    # 模型评估：输出遇难/幸存两类的查准率与召回率
from sklearn.model_selection import train_test_split # 数据集划分：分层抽样保持生死比例平衡
from sklearn.tree import DecisionTreeClassifier      # 核心算法：CART 决策树分类器 (支持 Gini 不纯度与 max_depth 剪枝)


def main() -> None:
    """泰坦尼克号生还预测主程序"""
    print("=" * 52)
    print("      🚢 决策树算法实战 —— 泰坦尼克号生还预测系统      ")
    print("=" * 52)

    # 【步骤 1】加载数据并完成特征预处理
    try:
        df = pd.read_csv("./datasets/train.csv")
    except Exception:
        print("⚠️ 数据集文件 ./datasets/train.csv 未找到，请检查数据路径。")
        return

    # 提取三大核心特征：舱位等级 (Pclass)、性别 (Sex)、年龄 (Age)
    X = df[["Pclass", "Sex", "Age"]].copy()
    
    # 缺失值填充：年龄缺失用全员均值填充
    X["Age"] = X["Age"].fillna(X["Age"].mean())
    
    # 类别变量编码 (如 Sex: male/female 转为 0/1)
    X = pd.get_dummies(X, drop_first=True)
    y = df["Survived"] # 0: 遇难, 1: 幸存

    print(f"📦 样本总量: {X.shape[0]} 名乘客 | 入模特征: {list(X.columns)}")

    # 【步骤 2】分层切分训练集 (80%) 与测试集 (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 【步骤 3】构建决策树并施加预剪枝 (max_depth=5)
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_clf.fit(X_train, y_train)

    # 【步骤 4】模型预测与全方位分类指标评估
    y_pred = tree_clf.predict(X_test)

    print("\n" + "=" * 52)
    print("📋 决策树生还分类评估报告 (max_depth=5 预剪枝):")
    print("-" * 52)
    print(classification_report(y_test, y_pred, target_names=["遇难 (Perished, 0)", "幸存 (Survived, 1)"]))
    print("=" * 52)


if __name__ == "__main__":
    main()
