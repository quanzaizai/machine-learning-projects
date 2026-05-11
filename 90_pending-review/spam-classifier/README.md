# spam-classifier

## 项目介绍
这是机器学习+NLP英文短信垃圾分类的小项目，这个项目代表了我开始接触这部分知识点并完成了对应的一个小项目。

## 当前功能
-  读取和准备数据
-  提取输入和标签，并划分训练集和测试集
-  文本向量化：将短信文本转换成 TF-IDF 数字特征
-  模型训练与评估
-  辅助查看数据和特征信息

## 运行方式
```
uv run main.py
```

## 运行示例
```
准确率：96.68%
特征矩阵大小：(4457, 7702)
   label                                               text
0      0  Go until jurong point, crazy.. Available only ...
1      0                      Ok lar... Joking wif u oni...
2      1  Free entry in 2 a wkly comp to win FA Cup fina...
3      0  U dun say so early hor... U c already then say...
4      0  Nah I don't think he goes to usf, he lives aro...
训练集大小：4457
测试集大小：1115
输入一条短信（q退出）：Free prize now, call this number!
这是垃圾短信
输入一条短信（q退出）：Are we still meeting tonight?
这是正常短信
输入一条短信（q退出）：q
```


## 当前学习重点
- pandas读取和准备数据
- 利用sklearn机器学习库的train_test_split来提取输入和标签，划分训练集和测试集
- 利用sklearn的TfidfVectorizer来将短信文本转换成数字特征的数据
- 利用sklearn的MultinomialNB进行模型的训练和评估
- 读取短信数据集属于获取数据，标签映射和数据划分属于数据处理，TF-IDF 文本向量化属于特征工程，MultinomialNB 的 fit() 属于模型训练，score() 属于模型评估，用户输入短信后 predict() 属于预测应用。