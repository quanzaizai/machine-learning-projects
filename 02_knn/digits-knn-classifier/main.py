"""
💡【知识点】KNN 手写数字识别 (8x8 灰度图像 0~9 多分类)
--------------------------------------------------------------------------------
📌【核心思想与本质】
  1. 图像特征：每个手写样本包含 64 个像素特征 (8x8 图像展开)。
  2. StandardScaler 标准化：将各像素点数值映射为均值 0、方差 1 的标准正态分布，消除特征尺度不一。
  3. GridSearchCV 自动化调参：3 折交叉验证寻找最优邻居数 K 与距离加权方式。
--------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def main():
    # ==================== 1. 数据加载与特征提取 ====================
    digits = load_digits()
    X = digits.data  # 1797 个样本，每个样本 64 维特征
    y = digits.target

    print(f"数据集规模: {X.shape[0]} 个样本，特征维度: {X.shape[1]}")

    # ==================== 2. 数据切分与标准化 ====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ==================== 3. 网格搜索调优 KNN ====================
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"]
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    test_acc = best_model.score(X_test, y_test)

    print(f"  • 最优超参数组合: {grid_search.best_params_}")
    print(f"  • 测试集识别准确率: {test_acc:.2%}")

if __name__ == "__main__":
    main()
