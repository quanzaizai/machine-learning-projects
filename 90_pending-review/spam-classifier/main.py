# 1. 读取和准备数据
import pandas as pd

df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "text"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 2. 提取输入和标签，并划分训练集和测试集
from sklearn.model_selection import train_test_split

X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 文本向量化：将短信文本转换成 TF-IDF 数字特征
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. 模型训练与评估
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

accuracy = model.score(X_test_tfidf, y_test)
print(f"准确率：{accuracy:.2%}")

# 5. 辅助查看数据和特征信息
print(f"特征矩阵大小：{X_train_tfidf.shape}")
print(df.head(5))
print(f"训练集大小：{len(X_train)}")
print(f"测试集大小：{len(X_test)}")

# 6. 用户输入预测
while True:
    text = input("输入一条短信（q退出）：")
    if text == "q":
        break

    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]

    if prediction == 1:
        print("这是垃圾短信")
    else:
        print("这是正常短信")