"""
💡【知识点】KNN 图像多分类实战 —— 8x8 手写数字识别与最小最大归一化
--------------------------------------------------------------------------------
📌【概念与本质】
  1. 图像特征展开：每个 8x8 灰度图像展开为 64 维像素值特征向量 (0 ~ 16)。
  2. 最小最大归一化 (MinMaxScaler)：将所有像素强度统一压缩至 [0, 1] 区间，公式：$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$。
  3. GridSearchCV 自动化搜索：在 10 分类任务中利用 5 折交叉验证遍历搜索 1 ~ 11 的最优 K 值。

📌【架构与模块分工】
  1. 数据加载与数据集切分 (load_digits, train_test_split)
  2. 特征工程：MinMaxScaler 像素归一化
  3. 模型构建与超参网格搜索 (GridSearchCV, cv=5)
  4. 测试集推断与指标报告
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
