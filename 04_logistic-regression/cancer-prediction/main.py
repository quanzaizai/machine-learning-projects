"""
【知识点】逻辑回归二分类 (乳腺癌良恶性预测)
--------------------------------------------------------------------------------
1. Sigmoid 函数：将线性输出映射到 (0, 1) 区间，转化为二分类概率。
2. 评估体系：混淆矩阵、精确率 (Precision)、召回率 (Recall) 与 F1-Score。
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
