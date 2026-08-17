"""
=============================================================================
💡【知识点】集成学习 (Ensemble Learning) —— 随机森林 (Random Forest) 与双重随机性
=============================================================================

📌【1. 随机森林 (Random Forest) 的核心哲学 —— Bagging 思想】
  - 集体智慧机制：单棵决策树容易过拟合（高方差），而由数百棵相互独立的决策树组成的“森林”
    通过**多数投票 (Majority Vote)** 能极大降低模型的整体方差，获得卓越的泛化性能！
  - **双重随机性保证独立性 (Double Randomness)** :
    ① **样本随机采样 (Bootstrap)** : 每次从训练集中有放回地随机抽取与原样本量相同数量的样本（约 63.2% 的样本会被选中，其余为袋外数据 OOB）。
    ② **特征随机抽样 (Feature Subsampling)** : 每个节点分裂时，只随机挑选部分特征（通常为 $\sqrt{M}$）来竞争最佳分裂点，防止强势特征垄断所有树。

📌【2. 超参数调优空间 (GridSearchCV)】
  - `n_estimators` : 森林中决策树的棵数（树越多越平稳，但计算耗时线性增加）。
  - `max_depth`    : 每棵树允许生长的最大深度（控制单树复杂度）。
=============================================================================
"""

# ==================== 0. Bagging 集成算法与网格搜索库引入 ====================
import pandas as pd                                  # 表格处理库：泰坦尼克号乘客特征清洗
from sklearn.ensemble import RandomForestClassifier  # 集成算法：随机森林分类器 (Bagging 重采样与特征子集随机化)
from sklearn.model_selection import GridSearchCV, train_test_split # 交叉验证网格超参数调优与数据集划分


def random_forest_classification() -> None:
    """随机森林建模与网格搜索调参主程序"""
    print("=" * 52)
    print("      🌲 集成学习 Bagging 典范 —— 随机森林 (Random Forest)      ")
    print("=" * 52)

    # 【步骤 1】加载数据并完成特征预处理
    try:
        df = pd.read_csv("./datasets/train.csv")
    except Exception:
        print("⚠️ 请确保 ./datasets/train.csv 存在。")
        return

    X = df[["Pclass", "Sex", "Age"]].copy()
    X["Age"] = X["Age"].fillna(X["Age"].mean())
    X = pd.get_dummies(X, drop_first=True)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=66, stratify=y
    )
    print(f"📦 训练集规模: {X_train.shape[0]} 例 | 测试集规模: {X_test.shape[0]} 例")

    # 【步骤 2】训练基线模型 (较浅的深度 max_depth=2 作为参照组)
    baseline = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=23)
    baseline.fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)
    print(f"🌱 基线模型测试准确率 (树棵数=100, max_depth=2): {baseline_acc:.2%}")

    # 【步骤 3】启动 3 折交叉验证网格搜索优化超参数
    param_grid = {
        "max_depth": [2, 3, 4, 5, 6],
        "n_estimators": [100, 200, 300]
    }
    print("\n⏳ 正在进行网格搜索交叉验证寻优...")
    gs = GridSearchCV(baseline, param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)

    # 【步骤 4】输出调优成果报告
    best_acc = gs.score(X_test, y_test)
    print("\n" + "=" * 52)
    print("🎯 网格调优成果总览:")
    print(f"  • 最优超参数组合 : {gs.best_params_}")
    print(f"  • 调优后测试准确率 : {best_acc:.2%} (提升: +{(best_acc - baseline_acc):.2%})")
    print("=" * 52)


if __name__ == "__main__":
    random_forest_classification()
