"""
=============================================================================
💡【知识点】逻辑回归二分类实战 —— 威斯康星乳腺癌良恶性智能诊断 (Medical Diagnosis)
=============================================================================

📌【1. 逻辑回归分类本质】
  - 线性组合映射到 Sigmoid 激活函数：
    $$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
  - 决策边界：若概率 $P \ge 0.5$ 判定为恶性 (Malignant)，否则判定为良性 (Benign)。

📌【2. 医疗诊断核心考点：为什么召回率 (Recall) 远比准确率更重要？】
  - 混淆矩阵模型：
    - **假阴性 (False Negative, 漏诊)**：病人明明是恶性肿瘤，模型却误判为“良性正常”！病人错过最佳治疗期，代价致命。
    - **假阳性 (False Positive, 误诊)**：病人其实是良性，模型误判为“恶性”。后续进一步复查可纠正，代价远小于漏诊。
  - **结论**：医疗诊断必须全力追求**高恶性召回率 (Recall = TP / (TP + FN))**！
=============================================================================
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    """乳腺癌良恶性分类主程序"""
    print("=" * 52)
    print("      🏥 逻辑回归二分类 —— 威斯康星乳腺癌辅助诊断系统      ")
    print("=" * 52)

    # 【步骤 1】加载数据并完成缺失值清洗
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

    X = df.iloc[:, 1:-1] # 剔除 Sample_code 纯 ID 列，提取 9 个细胞病理特征
    y = df["Class"]       # 标签: 2 代表良性 (Benign), 4 代表恶性 (Malignant)

    print(f"📦 样本总量: {X.shape[0]} 例病理记录 | 特征维度: {X.shape[1]} 维")

    # 【步骤 2】切分数据集 (75% 训练, 25% 测试)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 【步骤 3】特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 【步骤 4】逻辑回归建模与训练
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # 【步骤 5】多维度临床评估
    y_pred = clf.predict(X_test)

    print("\n" + "=" * 52)
    print("📋 临床多维度分类效能评估报告 (2:良性, 4:恶性):")
    print("-" * 52)
    print(classification_report(y_test, y_pred, target_names=["良性 (Benign)", "恶性 (Malignant)"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print("🔍 混淆矩阵 (Confusion Matrix):")
    print(f"  [实际良性] 预测良性(TN)={cm[0][0]:3d} | 预测恶性(FP)={cm[0][1]:3d}")
    print(f"  [实际恶性] 漏诊良性(FN)={cm[1][0]:3d} | 确诊恶性(TP)={cm[1][1]:3d}")
    print("=" * 52)


if __name__ == "__main__":
    main()
