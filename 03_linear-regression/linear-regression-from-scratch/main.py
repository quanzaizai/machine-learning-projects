import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


x = np.random.uniform(-5, 5, 100)
noise = np.random.normal(0, 1, 100)
w_true, b_true = 2, 3
y = w_true * x + b_true + noise
w, b = 0, 0
loss_history = []
lr = 0.01

for i in range(500):


#接着开始算预测的y
    y_pred = w*x + b

#开始算误差，预测减去真实值就等于误差
    error  = y_pred - y

#MSE就是误差平方后取平均
    loss = np.mean(error**2)
    loss_history.append(loss)

#接着这一步就是算梯度了，先算dw的，再算db的
    dw = np.mean((2*error * x))
    db = np.mean((2*error * 1))

#这一步通过学习率，然后更新参数w和b，那学习率在项目中好像也是自己定的，我记得最好是0.001到0.01之间是最好的，避免太大也避免太小
    w = w - lr * dw
    b  = b - lr * db

    print(f"损失函数为:{loss}")
    print(f"最新w参数为:{w}")
    print(f"最新b参数为:{b}")
#一轮线性回归就做完了，接下来需要不停的循环，把误差缩小，剩下的就是模型的训练，预测，评估

# 第一张：损失曲线
plt.plot(loss_history)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss curve')
plt.savefig('loss_curve.png', bbox_inches='tight')   # 先存
plt.show()                                                  # 再显示

# 第二张：拟合效果
plt.scatter(x, y, label='data')
x_line = np.linspace(-5, 5, 100)
plt.plot(x_line, w * x_line + b, color='red', label='fit')
plt.legend()
plt.savefig('fit.png', bbox_inches='tight')
plt.show()

#这边对比一下机器学习调用的API的结果
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
print("sklearn w:", model.coef_[0])
print("sklearn b:", model.intercept_)