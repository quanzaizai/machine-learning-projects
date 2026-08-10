"""
💡【知识点】决策树 (Decision Tree) 分类与泰坦尼克号生还预测
--------------------------------------------------------------------------------
📌【概念与本质】
  1. 决策树核心：通过基尼不纯度 (Gini) 或信息增益 (Information Gain) 递归选择最佳划分特征，构建树状判定逻辑。
  2. 预剪枝 (Pre-pruning)：通过设置 max_depth 限制树的最大生长深度，防止模型过度拟合训练集噪声。
  3. 树结构可视化 (plot_tree)：直观呈现每个决策节点的判定规则、样本量与主导类别。

📌【架构与模块分工】
  1. 特征选取与缺失值填补 (fillna with mean)
  2. 性别等类别特征独热编码 (pd.get_dummies)
  3. 分层数据集划分 (train_test_split, stratify=y)
  4. 决策树建模 (max_depth=10) 与分类报告
  5. 树状决策结构图绘制 (plot_tree)
--------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

def titanic_survival_prediction():
    # ==================== 1. 数据加载与特征选取 ====================
    data = pd.read_csv("./datasets/train.csv")
    
    # 选取关键特征与标签
    x = data[["Pclass", "Sex", "Age"]].copy()
    y = data["Survived"]

    # 缺失值均值插补
    x["Age"] = x["Age"].fillna(x["Age"].mean())
    
    # 类别特征独热编码 (Sex -> Sex_female, Sex_male)
    x = pd.get_dummies(x, columns=["Sex"])

    # ==================== 2. 分层数据集切分 ====================
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=66, stratify=y
    )

    # ==================== 3. 决策树建模与预剪枝 ====================
    estimator = DecisionTreeClassifier(max_depth=5, random_state=66)
    estimator.fit(X_train, y_train)

    # ==================== 4. 预测与评估 ====================
    y_pred = estimator.predict(X_test)
    print("=== 泰坦尼克号生还预测分类报告 ===")
    print(classification_report(y_test, y_pred, target_names=["未生还 (0)", "生还 (1)"]))

    # ==================== 5. 决策树结构可视化 ====================
    plt.figure(figsize=(20, 12))
    plot_tree(
        estimator,
        feature_names=x.columns,
        class_names=["未生还", "生还"],
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title("泰坦尼克号生还预测决策树结构", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    titanic_survival_prediction()
