import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

# 读取数据
test_data = pd.read_csv("./datasets/红酒品质分类_test.csv")
train_data = pd.read_csv("./datasets/红酒品质分类_train.csv")
X_train,y_train = train_data.iloc[:, :-1],train_data.iloc[:, -1]
X_test,y_test = test_data.iloc[:, :-1],test_data.iloc[: -1]

# 加载预训练模型
estimator = joblib.load("./datasets/红酒品质分类.pkl")

## 网格搜索 + 分层 K 折
param_dict = { "max_depth" :[1,2,3,4,5],
              "n_estimator":[100,200,300,400,500],
              "learning_rate" :[0.1,0.2,0.3,0.4,0.5]
            
}
#分类 + 不规律时使用的API
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=66)
gs = GridSearchCV(estimator,param_dict,cv=skf)
gs.fit(X_train,y_train)

print(f"最优模型为：{gs.best_params_}")
print(f"最优CV评分为:{gs.best_score_}")
y_pred = gs.predict(X_test)
print(f"测试集准确率:{accuracy_score(X_test,y_pred)}")