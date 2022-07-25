"""
Модуль текстовой классификации на основе архитектуры Transformer

Основан на https://keras.io/examples/nlp/text_classification_with_transformer/
"""

import json
import os

import numpy as np
from keras import layers
from keras.layers import TextVectorization
from ml.transformers import TransformerBlock, TokenAndPositionEmbedding
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras, string


def prepare_data(filename, tags: dict, category: str, train_size=0.8, test_size=0.2):
    """
    Функция загружает "сырые" текстовые данные и разделяет на обучающую
    и тестовую выборки.

    По умолчанию используется формат min-json из Label-studio.

    :param filename: файл с тренировочными данными в формате json
    :param tags: список категорий в формате dict
    :param category: LEVEL | AREA | STATUS
    :param train_size: доля обучающей выборки (от 0.0 до 1.0)
    :param test_size: доля тестовой выборки (от 0.0 до 1.0)
    :return: x_train, y_train, x_test, y_test
    """
    with open(filename, errors='ignore', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    print(f'Всего данных для обучения: {len(json_data)}')

    raw_text = [text.get('text', '') for text in json_data]
    raw_labels = [label.get(category, '') for label in json_data]

    """ Преобразуем категории в числовой вид """
    raw_labels = [tags.get(label, 0) for label in raw_labels]

    """ Разбиваем массив на обучающую и тестовую выборки в соотношении 80/20 """
    x_train, x_test, y_train, y_test = train_test_split(raw_text, raw_labels,
                                                        train_size=train_size, test_size=test_size)

    return x_train, y_train, x_test, y_test


class TextClassifierModel:
    """
    Модель текстовой классификации с использованием слоя Transformer.

    На выходе из слоя Transformer будет один вектор для каждого временного шага входной последовательности.
    Каждый раз берется среднее значение по всем временным шагам и
    используется сеть прямого распространения (FFN) для классификации текста.

    Модель распознает 3 "большие" категории: УРОВЕНЬ, ОТРАСЛЬ, СТАТУС.
    """

    """ [PAD] зарезервирован для дозаполнения (padding) текстовых данных """
    TAGS_LEVEL = {'[PAD]': 0, 'Муниципальный': 1, 'Региональный': 2, 'Федеральный': 3, 'Международный': 4}
    TAGS_AREA = {'[PAD]': 0, 'НацБез': 1, 'Госуправление': 2, 'Экономика': 3, 'Здравоохранение': 4, 'Транспорт': 5,
                 'НКО': 6, 'СоцПол': 7, 'Образование': 8, 'Прочее': 9}
    TAGS_STATUS = {'[PAD]': 0, 'Закон': 1, 'Новость': 2, 'Обновление': 3, 'Обзор': 4}

    NUM_TAGS_LEVEL = len(TAGS_LEVEL)
    NUM_TAGS_AREA = len(TAGS_AREA)
    NUM_TAGS_STATUS = len(TAGS_STATUS)

    def __init__(self, vocab_size=20000, max_len=2000, embed_dim=32, num_heads=2, ffn_dim=32, category='LEVEL'):
        super(TextClassifierModel, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.category = category
        if category == 'LEVEL':
            self.tags = self.TAGS_LEVEL
            self.num_tags = self.NUM_TAGS_LEVEL
        elif category == 'AREA':
            self.tags = self.TAGS_AREA
            self.num_tags = self.NUM_TAGS_AREA
        elif category == 'STATUS':
            self.tags = self.TAGS_STATUS
            self.num_tags = self.NUM_TAGS_STATUS
        self.tokenizer = TextVectorization(output_mode="int",
                                           output_sequence_length=self.max_len,
                                           max_tokens=self.vocab_size)
        self.tokenizer.adapt(['text for adapt'])

        # Start by creating an explicit input layer. It needs to have a shape of
        # (1,) (because we need to guarantee that there is exactly one string
        # input per batch), and the dtype needs to be 'string'.
        inputs = keras.Input(shape=(1,), dtype=string, name="text_layer")
        # The first layer in our model is the vectorization layer. After this layer,
        # we have a tensor of shape (batch_size, max_len) containing vocab indices.
        x = self.tokenizer(inputs)
        x = TokenAndPositionEmbedding(max_len=self.max_len, vocab_size=self.vocab_size, embed_dim=self.embed_dim)(x)
        x = TransformerBlock(self.embed_dim, self.num_heads, self.ffn_dim)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(self.num_tags, activation="softmax")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def train(self, batch_size=32, epochs=2, train_size=0.8, test_size=0.2):
        """
        Метод обучает модель и сохраняет отчёт по классификации категории

        :param batch_size: размер партии (части) данных в 1 итерации обучения
        :param epochs: число итераций (эпох) для обучения
        :param train_size: доля обучающей выборки (от 0.0 до 1.0)
        :param test_size: доля тестовой выборки (от 0.0 до 1.0)
        :return:
        """

        """ Загружаем обучающие данные """
        x_train, y_train, x_test, y_test = prepare_data(filename='./ml/data/train_set_1500.json',
                                                        tags=self.tags, category=self.category,
                                                        train_size=train_size, test_size=test_size)
        self.tokenizer.adapt(x_train)

        """ Строим и обучаем модель """
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        tb_callback = keras.callbacks.TensorBoard(f'./ml/data/logs/tcl/{self.category}/', update_freq='epoch')
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[tb_callback, es_callback]
        )
        os.makedirs('./ml/data/models', exist_ok=True)
        self.model.save(f'./ml/data/models/tcl_model_{self.category}_v1.tf', save_format='tf')

        """ Получение отчёта классификации """
        y_predicted = self.model.predict(x_test)
        y_predicted = list(np.argmax(y_predicted, axis=1))

        target_names = [i for i in self.tags.keys()]
        labels = [i for i in self.tags.values()]

        os.makedirs('./ml/data/results', exist_ok=True)
        with open(f'./ml/data/results/classification_report_for_{self.category}.txt', 'w', encoding='utf-8') as file:
            file.write(classification_report(y_test, y_predicted, target_names=target_names, labels=labels))

        with open(f'./ml/data/results/model_summary_for_{self.category}.txt', 'w', encoding='utf-8') as file:
            self.model.summary(print_fn=lambda x: file.write(x + '\n'))

    def load_model(self, category='LEVEL'):
        """
        Метод загружает обученную модель классификации одной из категорий:
        УРОВЕНЬ, ОТРАСЛЬ, СТАТУС

        :param category: LEVEL | AREA | STATUS
        :return:
        """
        """ Загружаем модель """
        if category == 'LEVEL':
            self.tags = self.TAGS_LEVEL
            self.num_tags = self.NUM_TAGS_LEVEL
            self.model = keras.models.load_model('./ml/data/models/tcl_model_LEVEL_v1.tf')
        elif category == 'AREA':
            self.tags = self.TAGS_AREA
            self.num_tags = self.NUM_TAGS_AREA
            self.model = keras.models.load_model('./ml/data/models/tcl_model_AREA_v1.tf')
        elif category == 'STATUS':
            self.tags = self.TAGS_STATUS
            self.num_tags = self.NUM_TAGS_STATUS
            self.model = keras.models.load_model('./ml/data/models/tcl_model_STATUS_v1.tf')
        else:
            raise ValueError("Category is not in list of ['LEVEL', 'AREA', 'STATUS']")

    def predict(self, text: str):
        """
        Метод возвращает название распознанной темы для категорий: УРОВЕНЬ, ОТРАСЛЬ, СТАТУС.

        :param text: слово, предложение, текст
        :return: {'category_name': 'category_name'}
        """
        y_predicted = self.model.predict([text])
        y_predicted = int(np.argmax(y_predicted))  # 1 число: индекс максимума

        # переводим число в название категории
        category_name = list(self.tags.keys())[list(self.tags.values()).index(y_predicted)]

        return {'category_name': category_name}
