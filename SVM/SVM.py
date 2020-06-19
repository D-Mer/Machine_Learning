import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.svm import LinearSVC


def load_data(dir):
    labels = []
    ham_files = []
    spam_files = []
    for enron in os.listdir(dir):
        sub_dir = os.path.join(dir, enron)
        if os.path.isdir(sub_dir):
            for d in os.listdir(sub_dir):
                if d == "ham":
                    for fi in os.listdir(os.path.join(sub_dir, "ham")):
                        ham_files.append(os.path.join(sub_dir, "ham", fi))
                        labels.append(0)
                elif d == "spam":
                    for fi in os.listdir(os.path.join(sub_dir, "spam")):
                        ham_files.append(os.path.join(sub_dir, "spam", fi))
                        labels.append(1)
    text_matrix = np.ndarray([len(ham_files) + len(spam_files)], dtype=object)
    index = 0
    for file in ham_files:
        with open(file, 'r', errors="ignore") as fi:
            next(fi)
            data = fi.read().replace('\n', ' ')
            text_matrix[index] = data
            index += 1
    for file in spam_files:
        with open(file, 'r', errors="ignore") as fi:
            next(fi)
            data = fi.read().replace('\n', ' ')
            text_matrix[index] = data
            index += 1
    return text_matrix, labels


if __name__ == '__main__':
    train_dir = "./train"
    train_matrix, train_labels = load_data(train_dir)
    print("train set size: ", train_matrix.shape[0])
    print("\tspam: ", train_labels.count(True))
    print("\tnon-spam: ", train_labels.count(False))

    count_v1 = CountVectorizer(stop_words="english", max_df=0.5, decode_error="ignore", binary=True)
    counts_train = count_v1.fit_transform(train_matrix)
    tfidftransformer = TfidfTransformer()
    tfidf_train = tfidftransformer.fit(counts_train).transform(counts_train)

    model = LinearSVC()
    model.fit(tfidf_train, train_labels)

    test_dir = "./test"
    test_matrix, test_labels = load_data(test_dir)
    print("test set size: ", test_matrix.shape[0])
    print("\tspam: ", test_labels.count(True))
    print("\tnon-spam: ", test_labels.count(False))

    count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_, stop_words="english", max_df=0.5, decode_error="ignore",
                               binary=True)
    counts_test = count_v2.fit_transform(test_matrix)
    tfidf_test = tfidftransformer.fit(counts_test).transform(counts_test)

    result = model.predict(tfidf_test)
    cm = pd.DataFrame(
        confusion_matrix(test_labels, result),
        index=["non-spam", "spam"],
        columns=["non-spam", "spam"]
    )

    print("predict result:")
    print(cm)
    print("precision socre: ", precision_score(test_labels, result))
    print("recall score: ", recall_score(test_labels, result))
    auc = roc_auc_score(test_labels, result)
    print("auc score: ", auc)

    FPR, recall, thresholds = roc_curve(test_labels, result, pos_label=1)

    # 画出ROC曲线
    plt.figure()
    plt.plot(FPR, recall, color='red'
             , label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    # 为了让曲线不黏在图的边缘
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
