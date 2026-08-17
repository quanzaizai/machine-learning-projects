"""
=============================================================================
💡【知识点】机器学习底层内功 —— 从零手撕线性回归 (Linear Regression from Scratch)
=============================================================================

📌【1. 线性回归数学模型全推导】
  - 假设函数 (Hypothesis) : $\hat{y} = Xw + b$
  - 均方误差损失函数 (MSE Loss) :
    $$J(w, b) = \frac{1}{2N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$
  - 梯度推导 (偏导数求导链式法则) :
    $$\frac{\partial J}{\partial w} = \frac{1}{N} X^T (\hat{y} - y)$$
    $$\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
  - 批量梯度下降更新公式 (Batch Gradient Descent) :
    $$w \leftarrow w - \alpha \cdot \frac{\partial J}{\partial w}$$
    $$b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}$$
    (其中 $\alpha$ 为学习率 Learning Rate)

📌【2. 梯度下降几何收敛模型】
  
  损失 Loss ^
            |  \  (初始随机点)
            |   \
            |    \ ---> 沿着负梯度最陡峭方向一步步迈向碗底
            |     \____
            |          \___ (全局最优极小点 min J(w,b))
            +----------------------------------> 迭代轮数 (Epochs)
=============================================================================
"""

# ==================== 0. 科学计算与类型系统模块引入 ====================
from typing import List, Tuple                       # 类型提示库：提供元组与列表强类型定义
import numpy as np                                   # 科学计算库：提供高性能矩阵点积 (dot)、转置 (T) 与广播机制


def generate_synthetic_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    生成带有高斯随机白噪声的真实线性回归数据
    真实模型方程: y = 3.0 * x + 4.0 + noise
    """
    np.random.seed(42)
    # 随机生成 100 个特征样本，范围在 [0, 2]
    X = 2 * np.random.rand(100, 1)
    # 按照真实系数生成目标标签，并加入标准差为 0.5 的高斯噪声
    y = 4.0 + 3.0 * X + np.random.randn(100, 1) * 0.5
    return X, y


def train_linear_regression(
    X: np.ndarray, 
    y: np.ndarray, 
    lr: float = 0.1, 
    epochs: int = 100
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    使用原生 NumPy 从零手写批量梯度下降算法 (BGD)

    :param X: 输入特征矩阵 (N, 1)
    :param y: 真实标签向量 (N, 1)
    :param lr: 学习率 (步长)
    :param epochs: 迭代总轮数
    :return: (最优权重 w, 最优偏置 b, 历史损失记录列表)
    """
    N = len(y)
    # 初始化权重 w (服从标准正态分布) 与偏置 b (全零)
    w = np.random.randn(1, 1)
    b = np.zeros((1, 1))
    loss_history: List[float] = []

    for epoch in range(epochs):
        # 【步骤 1】前向传播：计算预测值 y_pred = X * w + b
        y_pred = X.dot(w) + b

        # 【步骤 2】计算均方误差损失 MSE = (1 / 2N) * sum((y_pred - y)^2)
        loss = (1.0 / (2.0 * N)) * np.sum((y_pred - y) ** 2)
        loss_history.append(loss)

        # 【步骤 3】反向传播：根据解析梯度公式计算 dw 与 db
        dw = (1.0 / N) * X.T.dot(y_pred - y)
        db = (1.0 / N) * np.sum(y_pred - y)

        # 【步骤 4】参数更新：沿梯度的反方向迈进 lr 步长
        w -= lr * dw
        b -= lr * db

        if (epoch + 1) % 20 == 0:
            print(f"  [Epoch {epoch + 1:3d}/{epochs}] Loss: {loss:.4f} | w: {w[0][0]:.3f} | b: {b[0][0]:.3f}")

    return w, b, loss_history


def main() -> None:
    """手撕线性回归训练与验证主程序"""
    print("=" * 52)
    print("      📐 原生 NumPy 从零手撕线性回归 (BGD 梯度下降)      ")
    print("=" * 52)

    X, y = generate_synthetic_data()
    print("📊 成功生成 100 条合成数据样本 (真实公式: y = 3.0*x + 4.0 + ε)")
    print("⏳ 开始执行 100 轮批量梯度下降优化...\n")

    w, b, loss_history = train_linear_regression(X, y, lr=0.1, epochs=100)

    print("\n" + "=" * 52)
    print("🎯 训练完成与参数拟合对比:")
    print(f"  • 真实世界参数 : w = 3.000, b = 4.000")
    print(f"  • 模型拟合参数 : w = {w[0][0]:.3f}, b = {b[0][0]:.3f}")
    print(f"  • 初始损失 (Loss_0)   : {loss_history[0]:.4f}")
    print(f"  • 最终收敛损失 (Loss_end): {loss_history[-1]:.4f} (显著收敛！)")
    print("=" * 52)


if __name__ == "__main__":
    main()
