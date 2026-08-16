"""
💡【知识点】NLP 文本分类 —— TF-IDF 向量化与多项式朴素贝叶斯 (MultinomialNB)
--------------------------------------------------------------------------------
📌【核心思想与本质】
  1. TF-IDF：衡量词在短信中的出现频次与语料库逆文档频次，将变长文本矩阵化为特征向量。
  2. 朴素贝叶斯：基于词语条件独立性假设，通过贝叶斯定理计算文本属于垃圾短信 (Spam) 的后验概率。
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def main():
    try:
        df = pd.read_csv(
            "./90_pending-review/spam-classifier/SMSSpamCollection",
            sep="\t",
            names=["label", "message"]
        )
    except Exception:
        print("未找到 SMSSpamCollection 数据文件。")
        return

    # 1. 标签映射
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    X = df["message"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. TF-IDF 文本特征提取
    tfidf = TfidfVectorizer(stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 3. 朴素贝叶斯模型训练
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)

    y_pred = nb.predict(X_test_tfidf)
    print("=== 垃圾短信分类器 (MultinomialNB) 评估报告 ===")
    print(classification_report(y_test, y_pred, target_names=["正常短信 (Ham)", "垃圾短信 (Spam)"]))

if __name__ == "__main__":
    main()
