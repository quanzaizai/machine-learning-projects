import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 读取数据并打印数据集基本情况，并根据实际差异做出处理
df = pd.read_csv('./datasets/wine0501.csv')
df.info()
print(df.shape)
print(df.head())
print(df['Class label'].value_counts())
print(df.describe())

df = df[df['Class label']!= 1]#去掉1类
X = df[["Hue","Alcohol"]]#选取特征
y = df['Class label']#选取标签
y = LabelEncoder().fit_transform(df['Class label'])#将结果转成0/1

# 划分
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=66,
    stratify=y
)

# 建模
estimator = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3,random_state=66),
                               n_estimators=100,
                               learning_rate=0.1,
                               random_state=66)
estimator.fit(X_train,y_train)#训练
y_pred = estimator.predict(X_test)#预测
print(f"准确率为:{accuracy_score(y_test,y_pred)}")#评估