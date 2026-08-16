"""
【知识点】NLP 文本分类 —— TF-IDF 与朴素贝叶斯垃圾短信识别
--------------------------------------------------------------------------------
1. TF-IDF：将文本转换为词频-逆文档频率特征矩阵。
2. 朴素贝叶斯 (MultinomialNB)：利用贝叶斯定理快速计算垃圾短信的后验概率。
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def spam_classifier():
    # ==================== 1. 数据加载与标签二值化 ====================
    df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "text"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ==================== 2. TF-IDF 文本特征向量化 ====================
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ==================== 3. 朴素贝叶斯模型训练与评估 ====================
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    accuracy = model.score(X_test_tfidf, y_test)
    print("=== 朴素贝叶斯垃圾短信分类器 ===")
    print(f"语料库总样本: {len(df)} 条 (训练集: {len(X_train)}, 测试集: {len(X_test)})")
    print(f"TF-IDF 词表特征维度: {X_train_tfidf.shape[1]}")
    print(f"测试集分类准确率: {accuracy:.2%}\n")

    # ==================== 4. 示例预测与验证 ====================
    test_samples = [
        "Congratulations! You won a $1000 gift card. Call now to claim.",
        "Hey bro, are we still meeting for lunch at 12?"
    ]
    sample_tfidf = vectorizer.transform(test_samples)
    preds = model.predict(sample_tfidf)
    for sample, pred in zip(test_samples, preds):
        tag = "🚨 垃圾短信 (Spam)" if pred == 1 else "✅ 正常短信 (Ham)"
        print(f"文本: \"{sample}\" -> 预测结果: {tag}")

if __name__ == "__main__":
    spam_classifier()