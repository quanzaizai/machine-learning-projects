"""
💡【知识点】K-近邻算法 (KNN) 经典实践 —— 鸢尾花 (Iris) 分类与网格调优
--------------------------------------------------------------------------------
📌【概念与本质】
  1. KNN 核心原理：基于实例的学习（懒惰学习），通过计算待测样本与训练集中所有样本的欧氏距离，
     选取距离最近的 K 个邻居，采用“多数表决”决定分类类别。
  2. 特征标准化 (StandardScaler)：KNN 严重依赖距离度量，必须消除特征量纲差异（如厘米 vs 毫米）。
     - 训练集：transfer.fit_transform(X_train)（计算均值/方差并转换）。
     - 测试集：transfer.transform(X_test)（严格使用训练集的均值/方差转换，防止数据穿越）。
  3. 超参数搜索 (GridSearchCV)：结合 K-Fold 交叉验证自动寻找最优 K 值。

📌【架构与模块分工】
  1. 数据加载与探索 (iris_explore) : 打印特征、标签与形态。
  2. 数据集散点分布可视化 (iris_visualize) : Seaborn 绘制特征二维分布。
  3. 标准 KNN 建模流程 (iris_knn_pipeline) : 划分 -> 标准化 -> 建模 -> 评估。
  4. 网格搜索交叉验证调优 (iris_grid_search) : 自动搜索最优 K 值与评估。
--------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ==================== 1. 数据加载与探索 ====================

def iris_explore():
    """查看鸢尾花数据集的基本结构与特征描述"""
    iris = load_iris()
    print("=== 鸢尾花数据集概览 ===")
    print(f"特征名称: {iris.feature_names}")
    print(f"类别标签: {iris.target_names}")
    print(f"前5条样本特征:\n{iris.data[:5]}")
    print(f"前5条样本标签: {iris.target[:5]}\n")

# ==================== 2. 数据可视化 ====================

def iris_visualize():
    """基于花萼长宽绘制散点图查看样本分布"""
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df["label"] = iris.target
    
    plt.rcParams["font.family"] = ["Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    
    sns.lmplot(
        data=iris_df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        hue="label",
        fit_reg=False
    )
    plt.title("鸢尾花特征分布散点图", fontsize=12)
    plt.tight_layout()
    plt.show()

# ==================== 3. 标准 KNN 建模流水线 ====================

def iris_knn_pipeline():
    """完整的划分 -> 标准化 -> 训练 -> 评估流水线"""
    iris = load_iris()
    
    # 划分训练集与测试集 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=205611
    )

    # 特征标准化（消除尺度量纲差异）
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    # 创建 KNN 模型并训练 (指定 K=3)
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(X_train, y_train)

    # 模型预测与准确率评估
    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"=== 固定 K=3 模型准确率: {acc:.2%} ===")

# ==================== 4. 网格搜索与交叉验证自动调优 ====================

def iris_grid_search():
    """使用 GridSearchCV 自动搜索最优 K 值 (Hyperparameter Tuning)"""
    iris = load_iris()
    
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=66
    )

    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    # 待搜索候选 K 值列表 (通常取奇数避免平票)
    param_grid = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("=== 网格搜索与 5 折交叉验证结果 ===")
    print(f"最佳超参数 K 值: {grid_search.best_params_}")
    print(f"交叉验证最高得分: {grid_search.best_score_:.2%}")

    # 最终在独立测试集上考核最佳模型
    best_model = grid_search.best_estimator_
    test_acc = best_model.score(X_test, y_test)
    print(f"独立测试集最终准确率: {test_acc:.2%}\n")

# ==================== 5. 主执行入口 ====================

if __name__ == "__main__":
    iris_explore()
    iris_knn_pipeline()
    iris_grid_search()