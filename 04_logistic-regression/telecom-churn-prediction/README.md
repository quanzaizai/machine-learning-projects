# 电信客户流失预测

## 项目简介
这是逻辑回归这章节中的案例之一，这个案例能学习处理一些文本数据时的操作，主要还是得先学会看对应的数据集，然后根据具体的数据集来处理，在学习机器学习的时候得建立基本的操作流程。

## 使用技术
- pandas
- numpy
- sklearn中的:
    sklearn.model_selection的train_test_split
    sklearn.linear_model的LogisticRegression
    sklearn.metrics的accuracy_score, roc_auc_score, classification_report

## 项目功能
- 读取churn.csv数据集
- 把数据集中的字符串数据转换成对应的数字数据，同时去清理冗余的数据
- 手动确定特征和标签，接着去划分训练集和测试集，同时让标签的比率和划分前一致
- 进行逻辑回归的建模，然后用训练集来训练学习，测试集来预测
- 最后实现模型评估，用准确率和AUC这些作为指标,并打印出精确率、召回率、F1值等

## 运行结果和后续改进方案
运行结果:
```
dtypes: float64(2), int64(12), str(2)
memory usage: 880.5 KB
准确率为:0.7643718949609652
AUC为:0.79042470743238
              precision    recall  f1-score   support

          流失       0.81      0.89      0.85      1035
         未流失       0.58      0.43      0.49       374

    accuracy                           0.76      1409
   macro avg       0.69      0.66      0.67      1409
weighted avg       0.75      0.76      0.75      1409
```

后续改进方案:
- 代码层面可以修改为几个自定义函数
