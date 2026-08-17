"""
=============================================================================
💡【知识点】机器学习经典实战 —— KNN 手写数字高维图像识别 (Digits Classifier)
=============================================================================

📌【1. 图像多分类特征表示】
  - 每个样本由 8x8 像素的灰度矩阵展平为 **64 维特征向量** ($X \in \mathbb{R}^{1797 \times 64}$)。
  - 目标类别 $y \in \{0, 1, 2, \dots, 9\}$，属于经典的 10 分类模式识别问题。

📌【2. 距离加权与超参数自动化寻优 (GridSearchCV)】
  - `weights='uniform'`  : 邻居等权重投票。
  - `weights='distance'` : **反比距离加权**，越近的邻居对最终决策贡献越大（$w_i = 1 / d_i$）。
  - `GridSearchCV` (网格搜索交叉验证) :
    - 组合测试候选 $K \in [3, 5, 7, 9]$ 与不同的权重策略。
    - 结合 3 折交叉验证 (3-Fold CV)，避免在单一切分上因运气导致过拟合。
=============================================================================
"""

# ==================== 0. 核心算法与网格调参模块引入 ====================
from sklearn.datasets import load_digits             # 数据集库：加载 8x8 手写数字图像 (1797 例 64 维特征)
from sklearn.model_selection import GridSearchCV, train_test_split # 交叉验证网格搜索超参数寻优与数据集切分
from sklearn.neighbors import KNeighborsClassifier   # 核心算法：KNN 分类器 (支持 uniform 与 distance 加权)
from sklearn.preprocessing import StandardScaler     # 特征缩放：高维像素矩阵均值方差归一化


def main() -> None:
    """手写数字 KNN 识别与网格调参主程序"""
    print("=" * 52)
    print("      🔢 KNN 机器视觉实战 —— 8x8 手写数字图像识别      ")
    print("=" * 52)

    # 【步骤 1】加载手写数字图像数据集
    digits = load_digits()
    X = digits.data    # 1797 个样本，每个样本对应 64 维像素灰度值
    y = digits.target  # 0 ~ 9 的数字标签

    print(f"📦 数据集规模: {X.shape[0]} 例手写样本 | 特征维度: {X.shape[1]} (8x8 展开)")
    print(f"🏷️ 分类目标集: 0 ~ 9 共 10 个数字类别")

    # 【步骤 2】划分训练集 (80%) 与测试集 (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 【步骤 3】高维像素特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 【步骤 4】配置网格搜索超参数空间
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"]
    }
    
    print("\n⏳ 正在启动 3 折交叉验证网格搜索 (GridSearchCV)...")
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=3, n_jobs=-1, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    # 【步骤 5】提取最优模型并评估测试集
    best_model = grid_search.best_estimator_
    test_acc = best_model.score(X_test, y_test)

    print("\n" + "=" * 52)
    print("🎯 网格搜索与模型评估报告:")
    print(f"  • 最优超参数组合 : {grid_search.best_params_}")
    print(f"  • 交叉验证最佳得分 : {grid_search.best_score_:.2%}")
    print(f"  • 最终测试集准确率 : {test_acc:.2%}")
    print("=" * 52)


if __name__ == "__main__":
    main()
