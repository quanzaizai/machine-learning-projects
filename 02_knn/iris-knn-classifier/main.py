"""
=============================================================================
💡【知识点】机器学习经典算法 —— KNN 鸢尾花多分类 (K-Nearest Neighbors)
=============================================================================

📌【1. KNN 核心决策机制】
  - 核心原理：物以类聚，人以群分。
    - 给定一个未知种类的测试样本 $x$，计算它与训练集中所有已知样本之间的「欧几里得距离 (Euclidean Distance)」。
    - 挑选出距离最近的 $K$ 个邻居，通过「多数投票表决 (Majority Voting)」决定 $x$ 的类别归属。
  
📌【2. 距离度量与特征标准化 (StandardScaler) 的必要性】
  - 欧氏距离公式：$d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$
  - ⚠️ 关键陷阱：
    - 若特征 1（花萼长度）数值范围为 $[100, 1000]$，特征 2（花瓣宽度）数值范围为 $[0.1, 1.0]$。
    - 特征 1 的差值平方将彻底统治距离计算，导致特征 2 完全失效！
    - **标准化的本质**：$z = \frac{x - \mu}{\sigma}$，将所有特征统一缩放到均值为 0、方差为 1 的同一量纲下。

📌【3. 深度思考与高频 Q&A】

  ❓ Q1: 超参数 K 值的选取对模型有什么影响？
     👉 解答：
        - K 值过小（如 K=1）：模型过于敏感，容易受到噪声样本干扰发生「过拟合 (Overfitting)」。
        - K 值过大（如 K=100）：近邻中混入大量其他类别的样本，导致分类边界模糊发生「欠拟合 (Underfitting)」。
=============================================================================
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def main() -> None:
    """KNN 鸢尾花分类主流程"""
    print("=" * 48)
    print("      🌸 KNN 经典三分类 —— 鸢尾花种类识别      ")
    print("=" * 48)

    # 【步骤 1】加载经典 Iris 鸢尾花数据集 (150 个样本，4 个物理特征，3 类花卉)
    iris = load_iris()
    X, y = iris.data, iris.target
    print(f"📦 数据集总样本量: {X.shape[0]} 例 | 特征维度: {X.shape[1]} 维")
    print(f"🏷️ 目标分类品类: {list(iris.target_names)}")

    # 【步骤 2】划分训练集 (80%) 与测试集 (20%)
    # stratify=y 确保划分后训练集和测试集中各类别的比例与原数据集严格一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23, stratify=y
    )

    # 【步骤 3】特征标准化 (防止特征尺度不一主导欧氏距离)
    # ⚠️ 关键规范：只能用训练集 fit 统计均值和方差，测试集仅 transform，防止数据泄露！
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 【步骤 4】构建 KNN 模型并训练拟合
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # 【步骤 5】在独立测试集上评估分类泛化准确率
    accuracy = knn.score(X_test, y_test)
    print("\n" + "-" * 48)
    print(f"🎯 模型评估结果 (K={k} 近邻多数投票):")
    print(f"  • 测试集分类准确率 (Accuracy): {accuracy:.2%}")
    print("-" * 48)


if __name__ == "__main__":
    main()
