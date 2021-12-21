import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .processing import PreProcessText


class SentimentModel:
    """
    Class initialising simple sentiment model consisting of:

    1) Text preprocessing
    2) Vectorisation with TF-IDF
    3) Classification using Logistic Regression (LR)

    """

    def __init__(self, vocabulary, params):
        self.vocabulary = vocabulary
        self.params = params

    @property
    def _cfd(self):
        return os.path.dirname(os.path.realpath(__file__))

    @property
    def clf(self):
        return pickle.load(open(self._cfd + '/input/lr_model.pkl', 'rb'))

    @property
    def model_pipeline(self):
        return self.get_pipeline()

    @property
    def preprocess_pipeline(self):
        return self.get_pipeline(tokenize=True)[0]

    def get_pipeline(self, tokenize=False):
        """Returns preprocessing and vectorisation pipeline"""
        return Pipeline([
              ("preprocesser", PreProcessText(
                  lemmatize=self.params['preprocesser__lemmatize'],
                  remove_stopwords=self.params['preprocesser__remove_stopwords'],
                  remove_punctuation=self.params['preprocesser__remove_punctuation'],
                  to_lower=self.params['preprocesser__to_lower'],
                  remove_numbers=self.params['preprocesser__remove_numbers'],
                  min_length=int(self.params['preprocesser__min_length']),
                  tokenize=tokenize,
                  strip_html=True,
                  remove_url=True,
              )),
              ("vectorizer", TfidfVectorizer(
                  max_features=int(self.params['vectorizer__max_features']),
                  max_df=self.params['vectorizer__max_df'],
                  ngram_range=self.params['vectorizer__ngram_range'],
                  stop_words=self.params['vectorizer__stop_words'],
                  min_df=int(self.params['vectorizer__min_df']),
                  vocabulary=self.vocabulary
              )),
            ])

    def get_prediction_probs(self, df_text, human_readable=True):
        """Returns tuple with predicted probabilities for positive and negative sentiment for each entry in df_text"""
        embeddings = self.model_pipeline.fit_transform(df_text)
        prediction_probs = self.clf.predict_proba(embeddings)
        return self._transform_probs_to_dict(prediction_probs) if human_readable else prediction_probs

    def proc_in_vocabulary(self, df_text):
        """Returns list with proportion of words present in vocabulary for each entry in df_text"""
        df_tokenized = self.preprocess_pipeline.transform(df_text)
        return df_tokenized.apply(self._proc_text_in_vocabulary).tolist()

    @staticmethod
    def _transform_probs_to_dict(probs_array):
        """Returns tuple of dictionaries in which each dictionary contains probability text is positive or negative"""
        return tuple(map(lambda x: {'neg': x[0], 'pos': x[1]}, probs_array))

    def _proc_text_in_vocabulary(self, text_tokenized):
        """Returns proportion of words in text_tokenized also present in provided vocabulary"""
        return sum([1 for x in text_tokenized if x in self.vocabulary]) / len(text_tokenized)
