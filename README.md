# Machine Learning Projects (机器学习实践项目合集)

这是我的机器学习课程项目合集。目录按学习章节组织，每一章下面放对应的实践案例。目标是让“学过的知识点”和“写过的代码案例”保持一致：先按章节理解概念，再通过项目复现流程，最后把可运行代码沉淀下来。

## 🌟 项目亮点

- **按章分类**：完全匹配机器学习的学习路径，方便按图索骥复习知识点。
- **统一环境**：全仓库共用一个 Python 环境 (`pyproject.toml`)，告别重复安装依赖的烦恼。
- **数据解耦**：数据集统一放置在 `datasets/` 目录，代码逻辑更加清晰。

## 📁 目录结构与章节说明

| 章节目录 | 主题涵盖 | 核心实践项目 |
| --- | --- | --- |
| `01_ml-overview` | 机器学习整体框架和基础概念 | （理论积累） |
| `02_knn` | KNN 分类流程、距离度量、特征预处理等 | `iris-knn-classifier`、`digits-knn-classifier` |
| `03_linear-regression` | 线性回归、损失函数、梯度下降等 | `linear-regression-from-scratch` |
| `04_logistic-regression` | 逻辑回归、分类评估、ROC 与 AUC 等 | `cancer-prediction` |
| `90_pending-review` | 待归类/待复盘项目 | `spam-classifier` |

## 🚀 如何运行

本仓库使用一套共享的 Python 环境。推荐使用 `uv` 进行环境同步和运行。

```bash
# 1. 进入项目根目录并同步依赖环境
cd /Users/quanzaizai/Projects/machine-learning-projects
uv sync

# 2. 运行对应章节的具体项目案例
uv run python 04_logistic-regression/cancer-prediction/main.py
```
