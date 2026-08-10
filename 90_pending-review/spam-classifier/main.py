"""
💡【知识点】NLP 文本分类 —— TF-IDF 向量化与多项式朴素贝叶斯 (MultinomialNB) 垃圾短信过滤
--------------------------------------------------------------------------------
📌【概念与本质】
  1. TF-IDF (词频-逆文档频率)：
     - TF (Term Frequency)：词语在当前短信中的出现频次。
     - IDF (Inverse Document Frequency)：衡量词语在整个语料库中的常见度（越罕见越有区分度）。
     - 将变长文本矩阵化为定长数值特征向量。
  2. 朴素贝叶斯 (Naive Bayes)：基于条件独立性假设，通过贝叶斯定理 $P(C|X) = \frac{P(X|C)P(C)}{P(X)}$ 高效估算文本属于垃圾短信 (Spam) 的后验概率。

📌【架构与模块分工】
  1. 短信数据集读取与标签二值映射 (ham -> 0, spam -> 1)
  2. TfidfVectorizer 文本数字特征矩阵化
  3. MultinomialNB 朴素贝叶斯模型训练与准确率评估
  4. 交互式 CLI 实时短信预测
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