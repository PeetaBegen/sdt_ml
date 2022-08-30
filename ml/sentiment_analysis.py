"""
Модуль анализа тональности использует открытую библиотеку Dostoevsky.
https://github.com/bureaucratic-labs/dostoevsky

!!! Перед началом работы необходимо убедиться, что загружена модель !!!
Модель fasttext-social-network-model.bin должна быть в ./Lib/site-packages/dostoevsky/data/models/

Загрузка модели выполняется в консоли:
$ python -m dostoevsky download fasttext-social-network-model
"""

import torch
import translators as ts
from dostoevsky.models import FastTextSocialNetworkModel
from dostoevsky.tokenization import RegexTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, BertTokenizerFast


class SentimentAnalysisModel:
    RUBERT_DICT = {0: 'neutral', 1: 'positive', 2: 'negative'}

    def __init__(self, model='rubert'):
        """
        Инициализация класса SentimentAnalysisModel

        :param model: rubert | dostoevsky | nltk | siebert
        """
        super(SentimentAnalysisModel, self).__init__()
        self.tokenizer_dost = None
        self.model_dost = None
        self.tokenizer_rubert = None
        self.model_rubert = None
        self.model_nltk = None
        self.model_siebert = None
        self.is_eng = False

        if model == 'rubert':
            # Hugging Face -- blanchefort/rubert-base-cased-sentiment
            self.tokenizer_rubert = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment')
            self.model_rubert = AutoModelForSequenceClassification.from_pretrained(
                'blanchefort/rubert-base-cased-sentiment',
                return_dict=True)
        elif model == 'dostoevsky':
            # Dostoevsky
            self.tokenizer_dost = RegexTokenizer()
            self.model_dost = FastTextSocialNetworkModel(tokenizer=self.tokenizer_dost, lemmatize=True)
        elif model == 'nltk':
            # nltk -- VADER
            self.model_nltk = SentimentIntensityAnalyzer()
            self.is_eng = True
        elif model == 'siebert':
            # Hugging Face -- siebert/sentiment-roberta-large-english
            self.model_siebert = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
            self.is_eng = True
        else:
            raise NotImplementedError

    def predict(self, text: str):
        text_EN = 'NO TRANSLATION'
        sentiment_dost = None
        sentiment_dost_proba = None
        sentiment_nltk = None
        sentiment_siebert = None
        sentiment_rubert = None

        # переводим текст на английский, если is_eng=True
        if self.is_eng:
            text_EN = ts.bing(text, from_language='ru', to_language='en', if_use_cn_host=False)

            if self.model_nltk is not None:
                # используем nltk -- VADER
                sentiment_nltk = self.model_nltk.polarity_scores(text_EN)
                if sentiment_nltk.get('compound') >= 0.05:
                    sentiment_nltk = 'positive'
                elif sentiment_nltk.get('compound') <= -0.05:
                    sentiment_nltk = 'negative'
                else:
                    sentiment_nltk = 'neutral'

            if self.model_siebert is not None:
                # используем Hugging Face -- siebert/sentiment-roberta-large-english
                sentiment_siebert = str(self.model_siebert([text_EN])[0].get('label')).lower()

        else:
            if self.model_dost is not None:
                # Dostoevsky
                predictions = self.model_dost.predict([text], k=2)

                for i in predictions:  # type: dict
                    key, value = list(i.items())[0]  # 0 - выбор наиболее вероятного решения
                    sentiment_dost = key
                    sentiment_dost_proba = value

            if self.model_rubert is not None:
                # Hugging Face -- blanchefort/rubert-base-cased-sentiment
                @torch.no_grad()
                def predict_bert(text):
                    inputs = self.tokenizer_rubert(text, max_length=512, padding=True, truncation=True,
                                                   return_tensors='pt')
                    outputs = self.model_rubert(**inputs)
                    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
                    predicted = torch.argmax(predicted, dim=1).numpy()[0]  # 0 - выбор наиболее вероятного решения
                    predicted = self.RUBERT_DICT.get(predicted)
                    return predicted

                sentiment_rubert = predict_bert(text)

        return {'text_RU': text,
                'text_EN': text_EN,
                'sentiment': sentiment_rubert,  # основное значение тональности
                'sentiment_dost': sentiment_dost,
                'sentiment_dost_proba': sentiment_dost_proba,
                'sentiment_nltk': sentiment_nltk,
                'sentiment_siebert': sentiment_siebert,
                'sentiment_rubert': sentiment_rubert}
