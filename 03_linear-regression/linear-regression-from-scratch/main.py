"""
💡【知识点】线性回归从零手撕实现 (梯度下降优化法与损失收敛)
--------------------------------------------------------------------------------
📌【核心思想与本质】
  1. 预测模型：y_pred = w * x + b
  2. 损失函数 (MSE)：L = (1 / 2N) * sum((y_pred - y)^2)
  3. 梯度计算与参数更新：
     - dw = (1/N) * sum((y_pred - y) * x)
     - db = (1/N) * sum(y_pred - y)
     - w = w - lr * dw, b = b - lr * db
--------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np

def generate_synthetic_data():
    """生成带有高斯噪声的线性模拟数据"""
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1) * 0.5
    return X, y

def train_linear_regression(X, y, lr=0.1, epochs=100):
    """手动执行批量梯度下降 (BGD)"""
    N = len(y)
    w = np.random.randn(1, 1)
    b = np.zeros((1, 1))
    loss_history = []

    for epoch in range(epochs):
        # 1. 前向预测
        y_pred = X.dot(w) + b
        # 2. 计算均方误差损失
        loss = (1 / (2 * N)) * np.sum((y_pred - y) ** 2)
        loss_history.append(loss)

        # 3. 反向传播求梯度
        dw = (1 / N) * X.T.dot(y_pred - y)
        db = (1 / N) * np.sum(y_pred - y)

        # 4. 沿负梯度方向更新权重
        w -= lr * dw
        b -= lr * db

    return w, b, loss_history

def main():
    X, y = generate_synthetic_data()
    w, b, loss_history = train_linear_regression(X, y, lr=0.1, epochs=100)

    print("=== 从零手撕线性回归训练完成 ===")
    print(f"  • 真实参数: w=3.0, b=4.0")
    print(f"  • 拟合参数: w={w[0][0]:.3f}, b={b[0][0]:.3f}")
    print(f"  • 最终损失 (MSE): {loss_history[-1]:.4f}")

if __name__ == "__main__":
    main()
