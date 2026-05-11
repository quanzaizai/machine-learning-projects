# Datasets

课程里需要手动下载或导入的数据集统一放在这里。

建议用清晰的英文文件名，例如：

```text
datasets/breast_cancer.csv
datasets/house_prices.csv
```

代码里可以从仓库根目录读取：

```python
import pandas as pd

df = pd.read_csv("datasets/breast_cancer.csv")
```
