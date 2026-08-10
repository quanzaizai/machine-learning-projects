"""
💡【知识点】电信客户流失预测 —— 独热编码 (One-Hot)、分层采样与 ROC-AUC 评估
--------------------------------------------------------------------------------
📌【概念与本质】
  1. 类别特征独热编码 (pd.get_dummies)：将离散文本转为 0/1 哑变量，并删除冗余基准列避免共线性。
  2. 分层采样 (stratify=y)：针对客户流失等正负样本不均衡场景，确保训练集与测试集的正负样本比例严格一致。
  3. 综合评估指标：准确率 (Accuracy)、受试者工作特征曲线下面积 (ROC-AUC) 以及精确率/召回率报告 (classification_report)。

📌【架构与模块分工】
  1. 数据读取与哑变量转换 (pd.get_dummies, drop)
  2. 特征筛选与分层采样切分 (stratify=y)
  3. 逻辑回归拟合与多维度概率推断 (predict_proba)
  4. 全方位分类评估输出
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

def telecom_churn_prediction():
    # ==================== 1. 数据加载与独热编码 ====================
    churn_df = pd.read_csv("./datasets/churn.csv")
    churn_df = pd.get_dummies(churn_df)

    # 剔除冗余基准列与标签列
    churn_df.drop(["Churn_No", "gender_Male"], axis=1, inplace=True)
    churn_df.rename(columns={"Churn_Yes": "flag"}, inplace=True)

    # ==================== 2. 特征选择与分层划分 ====================
    X = churn_df[["Contract_Month", "internet_other", "PaymentElectronic"]]
    y = churn_df["flag"].astype(int)

    # 分层采样划分：保持流失/未流失原始比例
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=66, stratify=y
    )

    # ==================== 3. 逻辑回归建模与训练 ====================
    estimator = LogisticRegression(max_iter=1000)
    estimator.fit(X_train, y_train)

    # ==================== 4. 预测与全方位评估 ====================
    y_pred = estimator.predict(X_test)
    y_score = estimator.predict_proba(X_test)[:, 1] # 提取正类(流失)的预测概率

    print("=== 电信客户流失预测模型评估 ===")
    print(f"分类准确率 (Accuracy): {accuracy_score(y_test, y_pred):.2%}")
    print(f"ROC-AUC 得分:          {roc_auc_score(y_test, y_score):.4f}\n")
    
    print("=== 详细分类报告 (Precision / Recall / F1-Score) ===")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["未流失 (0)", "已流失 (1)"]))

if __name__ == "__main__":
    telecom_churn_prediction()
