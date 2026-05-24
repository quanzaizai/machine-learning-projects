import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 加载数据集
df = pd.read_csv("./datasets/train.csv")

#选特征 + 缺失值处理 + 独热编码处理
X = df[["Pclass","Sex","Age"]].copy()
X["Age"] = X["Age"].fillna(X["Age"].mean())
X = pd.get_dummies(X)
y = df["Survived"]

# 切分
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=66,
    stratify=y
)

# 建模
estimator = RandomForestClassifier(
    n_estimators=100,
    max_depth=2,
    random_state=23
)
estimator.fit(X_train,y_train)
print(f"随机森林的训练准确率为：{estimator.score(X_test,y_test)}")
#随机森林的训练准确率为：0.7988826815642458

# 利用网格搜索找最优超参
param_grid = {"max_depth":[1,2,3,4,5,6],"n_estimators":[100,200,300,400,500]}
gs = GridSearchCV(estimator,param_grid,cv=3)
gs.fit(X_train,y_train)
print(f"最优参数为：{gs.best_params_}")
print(f"调参后的准确率为：{gs.score(X_test,y_test)}")
#最优参数为：{'max_depth': 4, 'n_estimators': 500}
#调参后的准确率为：0.8156424581005587
