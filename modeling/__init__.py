import json
import os

from .model import SentimentModel

path_to_dir = os.path.dirname(os.path.realpath(__file__))

PARAMS = json.load(open(path_to_dir + '/input/best_params.json'))
VOCABULARY = json.load(open(path_to_dir + '/input/vocabulary.json'))

sentiment_model = SentimentModel(vocabulary=VOCABULARY, params=PARAMS)
