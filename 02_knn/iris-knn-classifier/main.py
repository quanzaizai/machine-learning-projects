"""
💡【知识点】KNN 经典三分类 —— 鸢尾花种类识别 (Iris Classification)
--------------------------------------------------------------------------------
📌【核心思想与本质】
  1. KNN 核心决策：计算测试样本与训练集中所有样本在 4 维特征空间中的欧氏距离，
     根据最近的 K 个邻居多数投票表决所属花卉品类。
  2. 标准化意义：花萼长度与花瓣宽度数值范围不同，标准化防止大数值特征主导距离计算。
--------------------------------------------------------------------------------
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def main():
    # ==================== 1. 加载数据集 ====================
    iris = load_iris()
    X, y = iris.data, iris.target

    # ==================== 2. 切分训练集与测试集 ====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23
    )

    # ==================== 3. 特征标准化 ====================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ==================== 4. KNN 模型训练与预测 ====================
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    accuracy = knn.score(X_test, y_test)
    print(f"=== 鸢尾花 KNN (K=5) 分类准确率: {accuracy:.2%} ===")

if __name__ == "__main__":
    main()
