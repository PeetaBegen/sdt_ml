"""
Основной модуль для сервиса машинного обучения.

Модуль использует обученные модели для задач текстовой классификации, анализа тональности,
распознавания именнованных сущностей и геокодирования.
"""

import pandas as pd
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
        with open(filename, errors='ignore', encoding='utf-8') as file:
            dfs = [pd.read_csv(file, delimiter=';')]

        p = pd.concat(dfs, ignore_index=True, sort=False)

        self.data = [str(item) for item in p['ТЕКСТ']]
        print(f'Загружено данных: {len(self.data)}')

    def make_predictions(self, file=None, text_list=None):
        """
        Метод использует обученные модели для задач Text Classification, NER, Sentiment Analysis
        и строит предсказания для каждой части текста

        :param file: CSV-файл для анализа (содержит обязательный столбец ТЕКСТ)
        :param text_list: массив текстов для анализа
        :return: JSON
        """
        if not file and not text_list:
            raise Exception('Данные для анализа отсутствуют. Передайте файл или массив текстов')
        if file:
            self.load_data(filename=file)
        else:
            self.data = text_list
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
        return {'status': 'Text classification is done', 'result': prediction}

    def train_text_classification(self):
        tcl_model = TextClassifierModel(vocab_size=20000, max_len=2000, embed_dim=32, num_heads=2,
                                        ffn_dim=32, category='STATUS')
        tcl_model.train(batch_size=8, epochs=10, train_size=0.8, test_size=0.2)

    def do_named_entity_recognition(self):
        text = 'Павел Петров и Лариса Мостовая решили усилить СМЭВ в 2022 году в Иркутске, ссылка http://www.irk.ru/'
        result = self.ner.do_slovnet_ner(text)  # Slovnet
        # self.ner.predicts(text)
        return {'status': 'NER prediction is done', 'result': result}

    def train_named_entity_recognition(self):
        ner_model = NERModel(vocab_size=20000, embed_dim=32, num_heads=4, ffn_dim=64)
        ner_model.train(batch_size=8, epochs=20)

    def do_sentiment_analysis(self):
        self.load_data(filename='./ml/data/NIRMA_comments_1.csv')
        result = [self.san.predict(text=i) for i in self.data]
        return {'status': 'Sentiment Analysis is done', 'result': result}

    def train_sentiment_analysis(self):
        pass

    def do_geocoding(self, query):
        return {'status': 'Geocoding is done', 'result:': self.geocoder.get_coordinates(query=query)}
