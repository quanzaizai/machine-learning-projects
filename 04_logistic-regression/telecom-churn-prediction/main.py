"""
💡【知识点】电信客户流失预测 —— 独热编码 (One-Hot)、分层采样与 ROC-AUC
--------------------------------------------------------------------------------
📌【核心思想与本质】
  1. 独热编码 (pd.get_dummies)：将字符串离散特征转为 0/1 哑变量。
  2. 分层采样 (stratify=y)：针对客户流失等正负样本不均衡场景，保证训练集和测试集流失比例一致。
  3. ROC-AUC：受试者工作特征曲线下面积，综合衡量分类器在不同阈值下的判别能力。
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

def main():
    try:
        df = pd.read_csv("./datasets/churn.csv")
    except Exception:
        print("数据集文件 ./datasets/churn.csv 未找到，请确保数据在 datasets 目录中。")
        return

    # 1. 类别特征独热编码
    df = pd.get_dummies(df, drop_first=True)

    target_col = [col for col in df.columns if "churn" in col.lower() or "class" in col.lower()][0]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. 分层采样划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 逻辑回归拟合
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("=== 电信客户流失预测报告 ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC 评分: {roc_auc_score(y_test, y_prob):.4f}")

if __name__ == "__main__":
    main()
