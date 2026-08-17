"""
=============================================================================
💡【知识点】分类综合实战 —— 电信客户流失预警 (Telecom Churn Prediction)
=============================================================================

📌【1. 核心技术栈与特征工程】
  - 类别特征独热编码 (One-Hot Encoding, `pd.get_dummies(..., drop_first=True)`) :
    - 将字符串离散特征（如“套餐类型”、“支付方式”）转化为二进制 0/1 哑变量。
    - `drop_first=True` 剔除第一个类别列，避免多重共线性（哑变量陷阱 Dummy Variable Trap）。
  - 分层采样 (Stratified Sampling, `stratify=y`) :
    - 针对客户流失等天然正负样本不均衡（流失客户往往只占 15%~20%）的场景，确保拆分后的训练集和测试集流失比例完全一致。
  - 综合评价指标 ROC-AUC (Area Under ROC Curve) :
    - 综合考察分类器在各种分类阈值（0.1~0.9）下的真阳率 (TPR) 与假阳率 (FPR) 平衡能力。
    - AUC 值越接近 1.0，代表模型的判别区分度越高。
=============================================================================
"""

# ==================== 0. 数据处理、逻辑回归与 ROC 评估库引入 ====================
import pandas as pd                                  # 数据分析库：提供 get_dummies 离散特征独热编码与表格清洗
from sklearn.linear_model import LogisticRegression   # 核心模型：对数几率回归分类器 (max_iter 迭代控制)
from sklearn.metrics import classification_report, roc_auc_score # 效能评估库：提供多维度分类报告与 ROC-AUC 判别得分计算
from sklearn.model_selection import train_test_split # 数据划分：基于流失分布的分层抽样 (stratify)


def main() -> None:
    """电信客户流失预测主程序"""
    print("=" * 52)
    print("      📞 电信客户流失智能预测与预警系统 (Churn)      ")
    print("=" * 52)

    # 【步骤 1】加载客户行为数据集
    try:
        df = pd.read_csv("./datasets/churn.csv")
    except Exception:
        print("⚠️ 数据集文件 ./datasets/churn.csv 未找到，请确保数据在 datasets 目录中。")
        return

    # 【步骤 2】特征工程：离散变量 One-Hot 编码 (防多重共线性)
    df = pd.get_dummies(df, drop_first=True)

    # 自动识别目标流失标签列
    target_col = [col for col in df.columns if "churn" in col.lower() or "class" in col.lower()][0]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"📦 数据规模: {X.shape[0]} 名客户 | 提取特征维度: {X.shape[1]} 维")

    # 【步骤 3】分层采样切分数据集 (80% 训练, 20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 【步骤 4】逻辑回归拟合 (设置 max_iter=1000 确保充分收敛)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 【步骤 5】计算离散预测结果与连续流失概率
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] # 提取判定为流失 (类别 1) 的预测概率

    # 【步骤 6】输出综合评估指标与 ROC-AUC 曲线评分
    auc_score = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 52)
    print("📋 客户流失预测模型效能报告:")
    print("-" * 52)
    print(classification_report(y_test, y_pred))
    print(f"🌟 综合判别指标 ROC-AUC 得分: {auc_score:.4f} (越接近 1.0 表现越优)")
    print("=" * 52)


if __name__ == "__main__":
    main()
