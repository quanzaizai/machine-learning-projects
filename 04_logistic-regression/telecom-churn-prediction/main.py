#导包
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report


#数据读取+从文本数据转向数字数据的处理
Churn_pd = pd.read_csv("./datasets/churn.csv")
Churn_pd.info()
Churn_pd = pd.get_dummies(Churn_pd)


#删掉冗余的特征列和标签列，因为get_dummies会把数据变成0和1的类别
Churn_pd.drop(["Churn_No","gender_Male"],axis=1,inplace=True)
Churn_pd.rename(columns={"Churn_Yes":"flag"},inplace=True)


#确定特征和标签
X = Churn_pd[["Contract_Month","internet_other","PaymentElectronic"]]
y = Churn_pd["flag"].astype(int)


#划分训练和测试集，stratify=y会让y保持划分前的比率
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=66,
    stratify=y
)


#模型训练
estimator = LogisticRegression(max_iter=1000)
estimator.fit(X_train,y_train)

#模型预测
y_perd = estimator.predict(X_test)
y_score = estimator.predict_proba(X_test)[:,1]

#模型评估
print(f"准确率为:{accuracy_score(y_test,y_perd)}")
print(f"AUC为:{roc_auc_score(y_test,y_score)}")
print(classification_report(y_test,y_perd,labels=[0,1],target_names=["流失","未流失"]))
