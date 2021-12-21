import re
import spacy
import string

from abc import ABC
from html.parser import HTMLParser
from io import StringIO
from itertools import compress
from sklearn.base import TransformerMixin


nlp = spacy.load("en_core_web_sm")


class HTMLStripper(HTMLParser, ABC):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self._text = StringIO()

    def handle_data(self, d):
        self._text.write(d)

    def get_data(self):
        return self._text.getvalue()


class PreProcessing:
    """Initialises preprocessing class with customizable text preprocessing steps"""

    def __init__(
            self, text,
            lemmatize=False,
            remove_stopwords=False,
            include_pos_tags=None,
            remove_punctuation=False,
            to_lower=False,
            remove_numbers=False,
            remove_url=False,
            min_length=0,
            strip_html=False,
            tokenize=False
    ):
        if include_pos_tags is None:
            include_pos_tags = []

        self._text = text
        self.include_pos_tags = include_pos_tags
        self.min_length = min_length
        self.is_tokenize = tokenize
        self.is_lemmatize = lemmatize
        self.is_remove_stopwords = remove_stopwords
        self.is_remove_punctuation = remove_punctuation
        self.is_to_lower = to_lower
        self.is_remove_numbers = remove_numbers
        self.is_remove_url = remove_url
        self.is_strip_html = strip_html

        self._token_dict = []
        self.mask = []

    @property
    def token_dict(self):
        if not self._token_dict:
            self.get_token_dict()
            self.mask = [True] * len(self._token_dict)
        return self._token_dict

    @property
    def text(self):
        return self._text.split() if self.is_tokenize else self._text

    def get_token_dict(self):
        """Retrieves Spacy token dictionary"""

        if type(self._text) == list:
            text = ''
        else:
            text = str(self.text)

        doc = nlp(text)
        self._token_dict = [{
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'tag': token.tag_,
            'is_stop': token.is_stop,
        } for token in doc]

    def lemmatize(self):
        """Applies lemmatization to given text (if necessary)"""
        if self.is_lemmatize:
            self._text = [token['lemma'] for token in self.token_dict]
        return self

    def remove_stopwords(self):
        """Removes English stopwords to given text (if necessary)"""
        if self.is_remove_stopwords:
            self.mask = [not self.token_dict[i]['is_stop'] * self.mask[i] for i in range(len(self.token_dict))]
        return self

    def remove_pos_tags(self, include_pos_tags):
        """Removes those positional words that are not part of include_pos_tags list"""
        if self.include_pos_tags:
            self.mask = [
                (self.token_dict[i]['pos'] in include_pos_tags) * self.mask[i] for i in range(len(self.token_dict))
            ]
        return self

    def apply_mask(self):
        """Combines lemmatization with removal of stopwords and excluded positional words"""
        if self.is_lemmatize or self.include_pos_tags or self.is_remove_stopwords:
            if type(self._text) != list:
                self._text = [token['text'] for token in self.token_dict]

            self._text = ' '.join(list(compress(self._text, self.mask)))
        return self

    def remove_punctuation(self):
        """Removes all punctuation from text"""
        if self.is_remove_punctuation:
            self._text = self._text.translate(str.maketrans('', '', string.punctuation))
        return self

    def to_lower(self):
        """Transforms text to lowercase"""
        if self.is_to_lower:
            self._text = self._text.lower() if type(self._text) == str else [token.lower() for token in self._text]
        return self

    def remove_numbers(self):
        """Removes any numbers from text"""
        if self.is_remove_numbers:
            self._text = self._text.translate(str.maketrans('', '', string.digits))
        return self

    def remove_url(self):
        """Removes any URL from text"""
        if self.is_remove_url:
            self._text = re.sub(r'http\S+', '', self._text)
        return self

    def remove_tiny_words(self, min_length):
        """Removes words from text small than or equal to min_length"""
        if self.min_length > 0:
            self._text = ' '.join(
                word for word in self._text.split() if len(word) > min_length or (word.isupper() and (2 < len(word) < 6))
            )
        return self

    def strip_html(self):
        """Strips any kind of HTML tags from text"""
        if self.is_strip_html:
            s = HTMLStripper()
            s.feed(self._text)
            self._text = s.get_data()
        return self

    def get_pos_tags(self):
        """Retries positional tags for each word in provided text"""
        return [token['pos'] for token in self.token_dict]

    def get_text_list(self):
        """Retrieves list of text"""
        return [token['text'] for token in self.token_dict]

    def parse(self):
        """Applies preprocessing pipeline to given text"""
        return (
            self
            .lemmatize()
            .remove_pos_tags(include_pos_tags=self.include_pos_tags)
            .remove_stopwords()
            .apply_mask()
            .strip_html()
            .remove_url()
            .remove_numbers()
            .remove_tiny_words(min_length=self.min_length)
            .remove_punctuation()
            .to_lower()
        ).text


class PreProcessText(TransformerMixin):
    """Initialises text preprocessing transformer to be used in Sklearn pipeline"""
    def __init__(self, **kwargs):
        self.params = kwargs

    def preprocess(self, text):
        return PreProcessing(text, **self.params).parse()

    def transform(self, df_text):
        return df_text.apply(self.preprocess)

    def fit(self, df_text, y=None):
        return self
