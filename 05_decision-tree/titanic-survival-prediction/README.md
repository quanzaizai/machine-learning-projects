# titanic-survival-prediction

## 项目简介
这是决策树的实战小案例，目的是更好的理解决策树的基本原理，包括对于基本基本API的运用。

## 使用技术
- pandas
- matplotlib
- sklearn.model_selection中的import train_test_split
- sklearn.tree中的DecisionTreeClassifier和plot_tree
- sklearn.metrics中的classification_report

## 项目功能
- 读取关于案例的数据集，并打印出数据集的基本数据
- 基于数据集，选择出对我们有用的特征和标签
- 处理选出标签或特征的缺失值，在这个案例中是处理Age这个特征，为了避免影响原样本，选择复制出副本来进行填充平均值
- 基于处理过后的数据集，划分训练集和测试集，同时保留标签在划分前的比率
- 利用决策树里的预剪枝，来降低后续过拟合的风险，建立决策树的分类模型后，进行训练-预测-评估
- 最后把整个处理过程可视化，并把特征和标签都按文字展示

## 运行结果
```
 11  Embarked     889 non-null    str    
dtypes: float64(2), int64(5), str(5)
memory usage: 83.7 KB
              precision    recall  f1-score   support

           0       0.88      0.83      0.85       110
           1       0.75      0.81      0.78        69

    accuracy                           0.82       179
   macro avg       0.81      0.82      0.81       179
weighted avg       0.83      0.82      0.82       179
```
