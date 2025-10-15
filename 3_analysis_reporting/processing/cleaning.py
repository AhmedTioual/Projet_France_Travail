from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import unicodedata
import re

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = text.lower()
            text = ''.join(
                c for c in unicodedata.normalize('NFD', text)
                if unicodedata.category(c) != 'Mn'
            )
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return X[self.key].apply(clean_text)