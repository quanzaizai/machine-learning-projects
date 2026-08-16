"""
【知识点】电信客户流失预测 (不均衡样本与 ROC-AUC)
--------------------------------------------------------------------------------
1. 独热编码 (One-Hot)：pd.get_dummies 将类别文本转换为 0/1 哑变量。
2. 分层采样 (stratify=y)：确保训练集与测试集的正负样本比例严格一致。
3. 综合评估：准确率结合 ROC-AUC 曲线下面积评估不均衡样本下的分类性能。
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
