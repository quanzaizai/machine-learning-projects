"""
【知识点】KNN 手写数字识别 (0~9 图像多分类)
--------------------------------------------------------------------------------
1. 数据集：load_digits() 包含 8x8 灰度像素手写数字图像 (1797 个样本)。
2. 特征工程：StandardScaler 标准化，消除不同像素点亮度的尺度差异。
3. 超参数调优：GridSearchCV 交叉验证寻找最优 K 近邻值与权重距离算法。
--------------------------------------------------------------------------------
"""

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

def digits_classification():
    # ==================== 1. 数据集加载与切分 ====================
    digits = load_digits()
    X = digits.data     # 64 维像素特征
    y = digits.target   # 数字类别 0 ~ 9

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=205611
    )

    # ==================== 2. 特征工程：像素归一化 ====================
    transfer = MinMaxScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    # ==================== 3. 网格搜索与交叉验证 ====================
    param_grid = {"n_neighbors": list(range(1, 12))}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # ==================== 4. 模型评估与前 10 样本预测 ====================
    y_pred = grid_search.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("=== 手写数字识别分类结果 ===")
    print(f"最佳超参数 K 值:   {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.2%}")
    print(f"测试集最终准确率: {test_acc:.2%}")
    print("-" * 35)
    print(f"前10个预测标签:   {y_pred[:10]}")
    print(f"前10个真实标签:   {y_test[:10]}")

if __name__ == "__main__":
    digits_classification()
