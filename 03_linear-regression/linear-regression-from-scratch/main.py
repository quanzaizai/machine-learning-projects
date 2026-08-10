"""
💡【知识点】从零手写一元线性回归 (Linear Regression) 与梯度下降算法 (Gradient Descent)
--------------------------------------------------------------------------------
📌【概念与数学本质】
  1. 假设函数 (Hypothesis)：$\hat{y} = w \cdot x + b$
  2. 损失函数 (MSE Loss)：$L(w, b) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 = \frac{1}{N} \sum_{i=1}^N (w x_i + b - y_i)^2$
  3. 梯度推导 (Gradients)：
     - 对权重的偏导：$\frac{\partial L}{\partial w} = \frac{2}{N} \sum (\hat{y}_i - y_i) \cdot x_i$
     - 对偏置的偏导：$\frac{\partial L}{\partial b} = \frac{2}{N} \sum (\hat{y}_i - y_i) \cdot 1$
  4. 参数更新规则 (Gradient Step)：$w = w - \alpha \cdot \frac{\partial L}{\partial w}, \quad b = b - \alpha \cdot \frac{\partial L}{\partial b}$

📌【架构与模块分工】
  1. 数据合成 (Data Synthesis) : 构造带高斯噪声的数据集 $y = 2x + 3 + \epsilon$。
  2. 梯度下降迭代引擎 (Training Loop) : 500 次前向预测、损失计算、反向梯度与参数更新。
  3. 训练过程与拟合效果可视化 : 导出 loss_curve.png 与 fit.png。
  4. 对标验证 (Benchmark) : 调用 sklearn 官方 LinearRegression 验证手动推导准确性。
--------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# ==================== 1. 数据合成与超参数初始化 ====================

np.random.seed(42)
x = np.random.uniform(-5, 5, 100)
noise = np.random.normal(0, 1, 100)

w_true, b_true = 2.0, 3.0
y = w_true * x + b_true + noise # 真实方程: y = 2x + 3 + noise

# 待学习参数初始化为 0
w, b = 0.0, 0.0
lr = 0.01          # 学习率 (Learning Rate)
epochs = 500       # 迭代轮数
loss_history = []  # 记录损失变化轨迹

# ==================== 2. 梯度下降核心训练循环 ====================

for epoch in range(epochs):
    # 步骤 ①：前向传播 (Forward Pass)
    y_pred = w * x + b
    
    # 步骤 ②：计算误差与均方误差损失 (MSE Loss)
    error = y_pred - y
    loss = np.mean(error ** 2)
    loss_history.append(loss)
    
    # 步骤 ③：反向传播计算梯度 (Compute Gradients)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error * 1)
    
    # 步骤 ④：沿梯度反方向更新参数 (Update Parameters)
    w = w - lr * dw
    b = b - lr * db

print("=== 从零手写梯度下降学习结果 ===")
print(f"迭代轮数: {epochs} 轮 | 最终 Loss: {loss:.4f}")
print(f"学得参数: w = {w:.4f} (真实值: {w_true}), b = {b:.4f} (真实值: {b_true})\n")

# ==================== 3. 结果可视化与图表存储 ====================

# 1. 绘制损失下降收敛曲线
plt.figure(figsize=(7, 4))
plt.plot(loss_history, color="blue", linewidth=1.5)
plt.xlabel("Iteration", fontsize=11)
plt.ylabel("MSE Loss", fontsize=11)
plt.title("Gradient Descent Loss Convergence Curve", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("loss_curve.png", bbox_inches="tight")
plt.close()

# 2. 绘制数据散点与回归拟合直线
plt.figure(figsize=(7, 4))
plt.scatter(x, y, color="steelblue", alpha=0.7, label="Sample Data")
x_line = np.linspace(-5, 5, 100)
plt.plot(x_line, w * x_line + b, color="crimson", linewidth=2, label=f"Fit Line (w={w:.2f}, b={b:.2f})")
plt.xlabel("x", fontsize=11)
plt.ylabel("y", fontsize=11)
plt.title("Linear Regression Fit", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("fit.png", bbox_inches="tight")
plt.close()

# ==================== 4. 对标 Scikit-Learn 官方结果 ====================

sk_model = LinearRegression()
sk_model.fit(x.reshape(-1, 1), y)
print("=== Scikit-Learn 官方库对标结果 ===")
print(f"Sklearn 计算得: w = {sk_model.coef_[0]:.4f}, b = {sk_model.intercept_:.4f}")