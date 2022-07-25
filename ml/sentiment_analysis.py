"""
Модуль анализа тональности использует открытую библиотеку Dostoevsky.
https://github.com/bureaucratic-labs/dostoevsky

!!! Перед началом работы необходимо убедиться, что загружена модель !!!
Модель fasttext-social-network-model.bin должна быть в ./Lib/site-packages/dostoevsky/data/models/

Загрузка модели выполняется в консоли:
$ python -m dostoevsky download fasttext-social-network-model
"""

from dostoevsky.models import FastTextSocialNetworkModel
from dostoevsky.tokenization import RegexTokenizer


class SentimentAnalysisModel:
    def __init__(self):
        self.tokenizer = RegexTokenizer()
        self.model = FastTextSocialNetworkModel(tokenizer=self.tokenizer, lemmatize=True)

    def predict(self, text: str):
        predictions = self.model.predict([text], k=2)
        sentiment = None  # значения тональности
        sentiment_proba = None  # вероятность тональности

        for i in predictions:  # type: dict
            key, value = list(i.items())[0]  # 0 - выбор наиболее вероятного решения
            sentiment = key
            sentiment_proba = value

        return {'text': text, 'sentiment': sentiment, 'sentiment_proba': sentiment_proba}
