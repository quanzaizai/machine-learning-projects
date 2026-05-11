# 1. 加载鸢尾花数据集
from sklearn.datasets import load_iris
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


# 1.查看数据集
def iris_1():
    iris = load_iris()
    print(f"鸢尾花的前五个特征:{iris.data[:5]}")
    print(f"鸢尾花的前五个标签:{iris.target[:5]}")
    print(f"鸢尾花数据的全部特征:{iris.feature_names}")
    print(f"鸢尾花数据的全部标签:{iris.target_names}")

# 2.数据集可视化
def iris_2():
    iris_1 = load_iris()
    iris_df = pd.DataFrame(iris_1.data,columns=iris_1.feature_names)
    iris_df["label"] = iris_1.target
    sns.lmplot(data=iris_df,x="sepal length (cm)",y="sepal width (cm)",hue="label",fit_reg=False)
    plt.title("iris_data")
    plt.tight_layout()
    plt.show()

# 3.定义函数，切分训练集和测试集
def iris_3():
    iris_1 = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
    iris_1.data,   
    iris_1.target,  
    test_size=0.2,    
    random_state=205611 
)
    print(f"训练集的题目数量: {X_train.shape[0]} 行")
    print(f"测试集的题目数量: {X_test.shape[0]} 行")

# 4.鸢尾花代码实现
def iris_4():
    #4.1接收鸢尾花数据
    iris_1 = load_iris()
    #4.2数据的预处理
    X_train,X_test,y_train,y_test = train_test_split(
        iris_1.data,
        iris_1.target,
        test_size=0.2,
        random_state=205611
    )
    #4.3创建标准化对象
    transfer = StandardScaler()
    #4.4对特征进行标准化
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)
    #4.5创建模型对象
    estimator = KNeighborsClassifier(n_neighbors=3)
    #4.6具体的训练模型
    estimator.fit(X_train,y_train)
    #4.7模型预测
    y_predict = estimator.predict(X_test)
    print(f"预测值为:{y_predict}")
    #4.8模型评估
    print(f"准确率：{estimator.score(X_test,y_test)}")
    print(f"准确率：{accuracy_score(y_test,y_predict)}")


# 5.K值不手动固定，用交叉验证和网格搜索寻找最优超参和最优评分
def iris_5():
    #接收鸢尾花数据
    iris_1 = load_iris()
    #数据预处理，切分成训练集和测试集
    X_train,X_test,y_tarin,y_test = train_test_split(
        iris_1.data,
        iris_1.target,
        test_size=0.2,
        random_state=66
    )
    #特征工程中的标准化
    transfer = StandardScaler()
    #训练集要学习
    X_train = transfer.fit_transform(X_train)
    #测试集用学习到的去测试，所以不加fit
    X_test = transfer.transform(X_test)
    #创建模型对象，并且准备开始用交叉验证和网格搜索
    estimator = KNeighborsClassifier()
    #定义要搜索的K值候选列表。
    param_grid = {'n_neighbors':[1,3,5,7,9,11]}
    grid_search = GridSearchCV(estimator,param_grid,cv=5)
    grid_search.fit(X_train,y_tarin)
    #找出最优超餐和最优评分
    print(f"找到的最佳参数：{grid_search.best_params_}")
    print(f"最优评分为：{grid_search.best_score_}")
    #创建最终的最优模型，然后拿着测试集去进行最终的考试
    best_model = grid_search.best_estimator_
    print(f"最终测试的成绩：{best_model.score(X_test,y_test)}")