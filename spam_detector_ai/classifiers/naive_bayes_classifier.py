# spam_detector_ai/classifiers/naive_bayes_classifier.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from .base_classifier import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        # CountVectorizer is used to convert the text data into a matrix of token counts
        self.vectoriser = CountVectorizer(**BaseClassifier.VECTORIZER_PARAMS)

    def train(self, X_train, y_train):
        # Convert the text data into a matrix of token counts
        X_train_vectorized = self.vectoriser.fit_transform(X_train).toarray()
        self.classifier = MultinomialNB()
        self.classifier.fit(X_train_vectorized, y_train)
