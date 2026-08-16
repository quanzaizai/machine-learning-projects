"""
💡【知识点】逻辑回归二分类 —— 威斯康星乳腺癌良恶性预测
--------------------------------------------------------------------------------
📌【核心思想与本质】
  1. Sigmoid 激活：将线性组合映射到 (0, 1) 区间，表示患病概率。
  2. 临床诊断考点：医学诊断中不仅看整体准确率，更看重**恶性肿瘤的召回率 (Recall)**，
     防止将恶性肿瘤误诊为良性（漏诊代价极高）。
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # ==================== 1. 数据加载与缺失值清洗 ====================
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    column_names = [
        "Sample_code", "Clump_Thickness", "Uniformity_Cell_Size", "Uniformity_Cell_Shape",
        "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei", "Bland_Chromatin",
        "Normal_Nucleoli", "Mitoses", "Class"
    ]
    try:
        df = pd.read_csv("./datasets/breast-cancer-wisconsin.csv")
    except Exception:
        df = pd.read_csv(url, names=column_names, na_values="?").dropna()

    X = df.iloc[:, 1:-1]
    y = df["Class"]

    # ==================== 2. 数据集切分与标准化 ====================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ==================== 3. 逻辑回归拟合与多维评估 ====================
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("=== 乳腺癌分类报告 (2:良性, 4:恶性) ===")
    print(classification_report(y_test, y_pred, target_names=["良性", "恶性"]))

if __name__ == "__main__":
    main()
