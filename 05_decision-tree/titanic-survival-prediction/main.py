import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report


#第一步 读取数据集
data = pd.read_csv("./datasets/train.csv")
data.info()

#第二步 数据预处理和特征工程
#2.1先选取特征和标签
x = data[["Pclass","Sex","Age"]]
y = data["Survived"]

#2.2处理Age的缺失值，为了避免影响到原来的data，先复制一份副本
x = x.copy()
x["Age"] = x['Age'].fillna(x["Age"].mean())

#2.3接下来对选出来的字符串数据Sex进行独热编码处理，转成0/1的数字列数据
x = pd.get_dummies(x,columns=["Sex"])

#第三步 划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=66,
    stratify=y
)

#第四步 进行建模和训练,并提前预剪枝，降低后续过拟合风险
estimator = DecisionTreeClassifier(max_depth=10)
estimator.fit(X_train,y_train)

#第五步 模型预测与评估
y_pred = estimator.predict(X_test)
print(classification_report(y_test,y_pred))

#第六步 可视化决策树
plt.figure(figsize=(30,20))
plot_tree(estimator,
          feature_names=x.columns,
          class_names=["未生还","生还"],
          filled=True,
          max_depth=10,
          fontsize=8)
plt.show()

