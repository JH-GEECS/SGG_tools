from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.sparse as sp
import joblib

import numpy as np

from tqdm import tqdm
import pandas as pd

import os
import glob


def csv_files_extractor(csv_dir):
    """
    여기서는 dataframe을 해체하여, subject, object를 추출하고
    이를 classifier가 학습할 수 있는 vector로서 반환한다.
    X, y를 반환한다.
    """
    X_subject_raw = []
    X_object_raw = []
    Y_predicate = []

    csv_list = glob.glob(os.path.join(csv_dir, '*.csv'))

    for csv_idx, each_csv in tqdm(enumerate(csv_list),
                                  total=len(csv_list),
                                  desc='csv files extractor'):
        each_df = pd.read_csv(each_csv)
        each_df_processed = each_df[each_df['rel_refined'] != 'Disposal']
        X_subject_raw.extend(each_df_processed['sclass'].tolist())
        X_object_raw.extend(each_df_processed['oclass'].tolist())
        Y_predicate.extend(each_df_processed['rel_refined'].tolist())

    vectorizer = CountVectorizer()
    X_subject = vectorizer.fit_transform(X_subject_raw)
    X_object = vectorizer.fit_transform(X_object_raw)
    min_features = X_object.shape[1] if X_object.shape[1] < X_subject.shape[1] else X_subject.shape[1]

    # redefine vectorizer, with smaller number of features
    vectorizer = CountVectorizer(max_features=min_features)
    X_subject = vectorizer.fit_transform(X_subject_raw)
    X_object = vectorizer.fit_transform(X_object_raw)

    X_tuple = sp.hstack([X_subject, X_object])

    # 여기서 vectorizer도 회수를 해야지, 새로운 data를 바로바로 integerize할 수 있다.
    return X_tuple, Y_predicate


class naive_bayes_classifier:
    def __init__(self, pretrained_model):
        if pretrained_model:
            self.classifier = joblib.load(pretrained_model)
        else:
            # laplace smoothing is implemented by default
            self.classifier = MultinomialNB(alpha=1.0)

    def train(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X, topk=None):
        if topk is None:
            return self.classifier.predict(X)
        else:
            y_probs = self.classifier.predict_proba(X)
            y_topk = np.argsort(y_probs, axis=1)[:, -topk:]
            # integerized word를 원래의 단어로 변경 필요
            itow_lambda = lambda x: self.classifier.classes_[x]
            return np.vectorize(itow_lambda)(y_topk)

    def save_model(self, save_path):
        joblib.dump(self.classifier, save_path)


if __name__ == "__main__":
    # 여기서는 예제 코드가 어떻게 동작하는 지 확인하고자 한다.
    """
    data = [("John likes pizza", "John", "pizza", "likes"),
            ("Mary hates spinach", "Mary", "spinach", "hates"),
            ("Bob loves sushi", "Bob", "sushi", "loves")]

    # Extract the subject and object from the data
    subjects = [d[1] for d in data]
    objects = [d[2] for d in data]
    labels = [d[3] for d in data]

    # Convert the subject and object into a vector of features
    vectorizer = CountVectorizer()
    X_subject = vectorizer.fit_transform(subjects)
    X_object = vectorizer.transform(objects)
    X = X_subject + X_object

    # Train the classifier
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, labels)
    """

    # 실제 활용할 때는 미리 write하고 predicate write할 때 중복확인하고 넣어주는 것으로 하자.

    csv_dir = r'E:\23.04.14\Files\whole_results_for_train'
    X, y = csv_files_extractor(csv_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = naive_bayes_classifier(pretrained_model=None)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    classifier.save_model(r'E:\23.04.14\Files\NB_classifier.joblib')