# 1. 加载手写数字识别的数据集
from sklearn.datasets import load_digits
# 2. 划分训练集和测试集
from sklearn.model_selection import train_test_split,GridSearchCV
# 3. 特征标准化
from sklearn.preprocessing import StandardScaler
# 4. KNN 分类模型
from sklearn.neighbors import KNeighborsClassifier
# 5. 模型评估
from sklearn.metrics import accuracy_score
# 6. 表格数据处理
import pandas as pd
# 7. 数据可视化
import matplotlib.pyplot as plt
# 8. 更高级的数据可视化库，基于 matplotlib
import seaborn as sns
# 9 数据预处理的归一化
from sklearn.preprocessing import MinMaxScaler

def digits_1():
    #这步就是先导入数据集
    digits = load_digits()
    X = digits.data
    y = digits.target
    #这步应该是要分割数据集了
    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=205611
    )
    #接下来数据处理,需要进行归一化操作
    transfer = MinMaxScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)
    #这一步是模型训练,我打算用交叉验证和网格搜索来解决，所以我没填具体的K值
    estimator = KNeighborsClassifier()
    prd = {"n_neighbors":[i for i in range(1,12)]}
    estimator_new = GridSearchCV(estimator,prd,cv=5)
    #接下来就是模型训练了,上面已经完成了交叉验证和网格搜索了,先训练这个训练集，训练集可以学习
    estimator_new.fit(X_train,y_train)
    #这一步应该做模型评估
    print(f"准确率为：{accuracy_score(y_test,estimator_new.predict(X_test))}")
    #这一步就做模型预测吧，模型保存先放着
    result = estimator_new.predict(X_test)
    print(f"预测值为：{result[:10]}")
    print(f"真实值为：{y_test[:10]}")
    print(f"最佳参数：{estimator_new.best_params_}")
    print(f"最佳交叉验证得分：{estimator_new.best_score_:.2%}")
    
digits_1()

    

