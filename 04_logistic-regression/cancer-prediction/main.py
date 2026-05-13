import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# 第一步：导入癌症数据
data = pd.read_csv("datasets/breast-cancer-wisconsin.csv")
data.info()


# 第二步：基本数据处理

# 2.1 处理异常值
# 把原数据中的 "?" 替换成真正的缺失值 np.nan，然后删除含缺失值的行
data = data.replace(to_replace="?", value=np.nan)
data = data.dropna()


# 2.2 确定特征值和标签值
# X 是特征值，也就是模型用来判断癌症类别的输入数据
# y 是标签值，也就是真实类别 Class
X = data.iloc[:, 1:-1].astype(float)
y = data["Class"]


# 2.3 分割训练集和测试集
# 训练集用来训练模型，测试集用来检验模型效果
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=66
)


# 第三步：特征工程
# 标准化：把不同特征缩放到相近尺度，避免某些数值范围大的特征过度影响模型
transfer = StandardScaler()

# 训练集：fit_transform，先学习训练集的均值/标准差，再标准化训练集
X_train = transfer.fit_transform(X_train)

# 测试集：只能 transform，使用训练集学到的均值/标准差来标准化测试集
X_test = transfer.transform(X_test)


# 第四步：模型训练
# 癌症预测是分类任务，所以这里使用逻辑回归分类模型
estimator = LogisticRegression()
estimator.fit(X_train, y_train)


# 第五步：模型评估

# 5.1 预测测试集结果
y_pred = estimator.predict(X_test)
print(f"y的预测值为: {y_pred}")


# 5.2 计算分类准确率
accuracy = estimator.score(X_test, y_test)
print(f"准确率为: {accuracy}")