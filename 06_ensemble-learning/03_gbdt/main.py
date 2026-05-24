import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# 读取数据
df = pd.read_csv("./datasets/train.csv")

# 数据预处理
X = df[["Pclass","Sex","Age"]]#选特征
y = df["Survived"]#选标签
X = X.copy()#复制副本
X["Age"] = X["Age"].fillna(X["Age"].mean())#填充平均值
X = pd.get_dummies(X)#进行独热编码处理

# 划分
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=66,
    stratify=y
)

# 建模
estimator = GradientBoostingClassifier(n_estimators=100,
                                       learning_rate=0.1,
                                       random_state=66)
estimator.fit(X_train,y_train)
print(f"准确率为：{estimator.score(X_test,y_test)}")

# 网格搜索来寻找最优参数
param_grid = {"learning_rate":[0.1,0.2,0.3,0.4,0.5],'n_estimators':[100,200,300,400,500]}
gs = GridSearchCV(estimator,param_grid,cv=3)
gs.fit(X_train,y_train)
print(f"最优模型为：{gs.best_estimator_}")
print(f"准确率为:{gs.score(X_test,y_test)}")