"""
💡【知识点】逻辑回归 (Logistic Regression) 二分类 —— 乳腺癌良恶性预测
--------------------------------------------------------------------------------
📌【概念与本质】
  1. 逻辑回归本质：通过 Sigmoid 激活函数 $g(z) = \frac{1}{1 + e^{-z}}$ 将线性回归输出映射到 $(0, 1)$ 概率区间。
  2. 缺失值清洗：原始数据中以字符串 "?" 充当缺失标记，必须先替换为 np.nan，再执行 dropna() 清洗。
  3. 标准化一致性：测试集严格使用训练集学得的均值与方差 (transform) 避免数据穿越 (Data Leakage)。

📌【架构与模块分工】
  1. 数据加载与缺失值清洗 (read_csv, replace, dropna)
  2. 特征矩阵 X 与标签 y 提取及数据集切分 (train_test_split)
  3. 特征标准化 (StandardScaler)
  4. 逻辑回归建模训练与预测评估 (LogisticRegression, score)
--------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cancer_prediction():
    # ==================== 1. 数据加载与缺失值清洗 ====================
    data = pd.read_csv("datasets/breast-cancer-wisconsin.csv")
    
    # 替换 "?" 为 np.nan 并剔除空值行
    data = data.replace(to_replace="?", value=np.nan).dropna()

    # ==================== 2. 特征与标签提取与切分 ====================
    # 排除第一列 ID 与最后一列类别 Class
    X = data.iloc[:, 1:-1].astype(float)
    y = data["Class"] # 2 为良性，4 为恶性

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=66
    )

    # ==================== 3. 特征标准化 ====================
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    # ==================== 4. 逻辑回归建模与训练 ====================
    estimator = LogisticRegression()
    estimator.fit(X_train, y_train)

    # ==================== 5. 预测与评估 ====================
    y_pred = estimator.predict(X_test)
    accuracy = estimator.score(X_test, y_test)

    print("=== 乳腺癌良恶性预测结果 ===")
    print(f"前 10 个测试样本预测: {y_pred[:10]}")
    print(f"前 10 个测试样本真实: {y_test.values[:10]}")
    print(f"模型分类准确率 (Accuracy): {accuracy:.2%}")

if __name__ == "__main__":
    cancer_prediction()
