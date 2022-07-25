"""
Основной модуль для сервиса машинного обучения.

Модуль использует обученные модели для задач текстовой классификации, анализа тональности,
распознавания именнованных сущностей и геокодирования.
"""

import pandas as pd
from sdt_ml.settings import DATA_DIR
from ml.geocoder import GeocoderOSM
from ml.ner import NERModel
from ml.sentiment_analysis import SentimentAnalysisModel
from ml.text_classification import TextClassifierModel


class MLService:
    def __init__(self):
        self.tcl_level = TextClassifierModel()
        self.tcl_level.load_model(category='LEVEL')

        self.tcl_area = TextClassifierModel()
        self.tcl_area.load_model(category='AREA')

        self.tcl_status = TextClassifierModel()
        self.tcl_status.load_model(category='STATUS')

        self.ner = NERModel()
        self.ner.load_model()

        self.san = SentimentAnalysisModel()

        self.geocoder = GeocoderOSM()

        self.data = []

    def load_data(self, filename):
        # По умолчанию filename='./ml/data/test_example.csv'
        with open(filename, errors='ignore', encoding='utf-8') as file:
            dfs = [pd.read_csv(file, delimiter=';')]

        p = pd.concat(dfs, ignore_index=True, sort=False)

        self.data = [str(item) for item in p['ТЕКСТ']]
        print(f'Загружено данных: {len(self.data)}')

    def make_predictions(self, file='./ml/data/test_example.csv'):
        """
        Метод использует обученные модели для задач Text Classification, NER, Sentiment Analysis
        и строит предсказания для каждой части текста

        :param file: CSV-файл для анализа (содержит обязательный столбец ТЕКСТ)
        :return: JSON
        """
        self.load_data(filename=file)
        data = {"data": []}
        for text in self.data:
            data['data'].append({"text": text,
                                 "sentiment": self.san.predict(text).get('sentiment'),
                                 "area": self.tcl_area.predict(text).get('category_name'),
                                 "level": self.tcl_level.predict(text).get('category_name'),
                                 "status": self.tcl_status.predict(text).get('category_name'),
                                 "ner": self.ner.do_slovnet_ner(text)
                                 })
        return data

    def do_text_classification(self):
        text = 'Обсуждаем новости о международным уровне в госуправлении'
        prediction = self.tcl_level.predict(text)
        return prediction

    def train_text_classification(self):
        self.tcl_level = TextClassifierModel(vocab_size=20000, max_len=2000, embed_dim=32, num_heads=2,
                                             ffn_dim=32, category='STATUS')
        self.tcl_level.train(batch_size=8, epochs=3, train_size=0.3, test_size=0.1)

    def do_named_entity_recognition(self):
        text = 'Павел Петров решил усилить СМЭВ в 2022 году в Иркутске'
        self.ner.do_slovnet_ner(text)  # Slovnet
        self.ner.predicts(text)
        return {'ner': 'NER prediction is done'}

    def train_named_entity_recognition(self):
        self.ner = NERModel(vocab_size=20000, embed_dim=32, num_heads=4, ffn_dim=64)
        self.ner.train(batch_size=8, epochs=3)

    def do_sentiment_analysis(self):
        self.load_data(filename='./ml/data/test_example.csv')
        return self.san.predict(text=self.data[0])

    def train_sentiment_analysis(self):
        pass

    def do_geocoding(self, query):
        return self.geocoder.get_coordinates(query=query)
