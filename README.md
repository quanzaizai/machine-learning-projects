# Machine Learning Projects (机器学习经典算法实战)

本项目为机器学习经典算法的理论推导、从零实现与工程实战案例合集。

---

## ⚡ 快速环境配置

本项目统一使用 `uv` 管理虚拟环境与依赖：

```bash
# 同步安装科学计算与机器学习依赖
uv sync

# 运行任意案例（以随机森林调参为例）
uv run python 06_ensemble-learning/01_random-forest/main.py
```

---

## 📚 算法章节与中文导读索引

| 章节与目录 | 中文算法/案例名 | 核心知识点与实现机制 | 数据集 |
| :--- | :--- | :--- | :--- |
| **`01_ml-overview/`** | **机器学习全景概览** | 有监督 vs 无监督学习、特征工程生命周期与评估指标体系 | - |
| **`02_knn/`** | **K-近邻分类器 (KNN)** | 距离度量 (欧氏距离)、K 值选择、鸢尾花与手写数字识别 | `iris`, `digits` |
| **`03_linear-regression/`** | **线性回归从零手撕** | 梯度下降优化算法 (BGD/SGD)、损失函数收敛曲线绘制 | 仿真回归数据 |
| **`04_logistic-regression/`** | **逻辑回归与二分类** | Sigmoid 函数、对数损失、电信客户流失预测与癌症良恶性诊断 | `churn.csv`, `breast-cancer.csv` |
| **`05_decision-tree/`** | **决策树算法实战** | 信息增益 (ID3)、基尼系数 (CART)、泰坦尼克号生存预测 | `titanic` |
| **`06_ensemble-learning/`** | **集成学习四大名捕** | 随机森林 (Bagging)、AdaBoost、GBDT 与 XGBoost 调优对比 | `train.csv` |
| **`07_kmeans-clustering/`** | **K-Means 聚类** | 无监督聚类、肘部法则 (Elbow Method)、轮廓系数评估 | `wine0501.csv` |
| **`90_pending-review/`** | **待复核/练习案例** | 贝叶斯垃圾短信分类器 (SMS Spam Collection) | `SMSSpamCollection` |
