"""
Модуль распознавания именнованых сущностей (NER) на основе архитектуры Transformer.

Основан на https://keras.io/examples/nlp/ner_transformers/
"""

from keras import layers
from ml.transformers import TransformerBlock, TokenAndPositionEmbedding
from tensorflow import keras
import os
import json
import pickle
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from collections import Counter
from ml.conlleval import evaluate
import pymorphy2
from ml.geocoder import GeocoderOSM
import re
import pyconll
from sklearn.metrics import classification_report
from navec import Navec
from slovnet import NER
from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,

    Doc
)


def export_to_txt_file(export_file_path, data: list):
    """
    Функция сохраняет данные в .txt

    :param export_file_path: Имя выходного файла
    :param data: list of lists
    :return:
    """
    with open(export_file_path, "w", encoding='utf-8') as f:
        # record[0] - tokens
        # record[1] - tags
        for record in data:
            if len(record[0]) > 0:
                f.write(
                    str(len(record[0]))
                    + "\t"
                    + "\t".join(record[0])
                    + "\t"
                    + "\t".join(map(str, record[1]))
                    + "\n"
                )


def read_conll_2003(filename):
    """
    Функция для считывания файла в формате CoNLL-2003.

    Формирует пары из токена (слова) и метки, например ['Организация','B-ОРГАНИЗАЦИЯ']

    :param filename: Файл CoNLL-2003
    :return: ([['word_1','NER_tag_1'],['word_2', 'NER_tag_2']])
    """
    with open(filename, errors='ignore', encoding='utf-8') as file:
        split_labeled_text = []
        sentence = []
        for line in file:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    split_labeled_text.append(sentence)
                    sentence = []
                continue
            splits = line.split(' ')
            sentence.append([splits[0], splits[-1].rstrip("\n")])

        if len(sentence) > 0:
            split_labeled_text.append(sentence)
            sentence = []
        return tuple(split_labeled_text)


def get_tokens_and_tags(dataset, ner_tags: dict):
    """
    Функция получает все токены, переводит метки в числа, получает размерность вектора

    :param dataset: датасет из CoNLL
    :param ner_tags: словарь из меток для NER
    :return: (new_dataset, all_tokens, max_len)
    """
    # fetching all tokens and tags and counting max_len
    all_tokens = []
    tokens = []
    tags = []
    new_dataset = []
    max_len = 0
    for sentence in dataset:
        for token in sentence:
            all_tokens.append(token[0])
            tokens.append(token[0])
            tags.append(list(ner_tags.keys())[list(ner_tags.values()).index(token[1])])
        new_dataset.append(([*tokens], [*tags]))
        if len(tokens) > max_len:
            max_len = len(tokens)
        tokens = []
        tags = []
    return new_dataset, all_tokens, max_len


def map_record_to_training_data(record):
    """
    Функция переводит данные в формат tf.data.Dataset

    :param record: запись из файла .txt (CoNLL-2003)
    :return: (tokens, tags)
    """
    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1: length + 1]
    tags = record[length + 1:]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    return tokens, tags


def make_tag_lookup_table():
    """
    Функция формирует и возвращает словарь сущностей

    :return: {0:'[PAD]', 1:'O', ... , n:'entity'}
    """
    iob_labels = ['B', 'I']
    ner_labels = ['ОРГАНИЗАЦИЯ', 'ТЕХНОЛОГИЯ', 'УЧАСТНИК', 'ГЕОГРАФИЯ', 'URL']
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ['-'.join([a, b]) for a, b in all_labels]
    all_labels = ['[PAD]', 'O'] + all_labels
    return dict(zip(range(0, len(all_labels) + 1), all_labels))


class NERModel(keras.Model):
    NER_TAGS = make_tag_lookup_table()  # type: dict
    NUM_TAGS = len(NER_TAGS)
    MORPH = pymorphy2.MorphAnalyzer()
    NAVEC = Navec.load('./ml/data/navec_news_v1_1B_250K_300d_100q.tar')
    NER_SLOVNET = NER.load('./ml/data/slovnet_ner_news_v1.tar')
    NER_SLOVNET.navec(NAVEC)
    GEOCODER = GeocoderOSM()

    def __init__(self, vocab_size=20000, max_len=128, embed_dim=32, num_heads=2, ffn_dim=32, vocabulary=None):
        super(NERModel, self).__init__()
        self.num_tags = self.NUM_TAGS
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.embedding_layer = TokenAndPositionEmbedding(self.max_len, self.vocab_size, self.embed_dim)
        self.transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.ffn_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ffn = layers.Dense(self.ffn_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ffn_final = layers.Dense(self.num_tags, activation="softmax")

        if vocabulary is None:
            vocabulary = []
        self.vocabulary = vocabulary
        self.lookup_layer = keras.layers.StringLookup()

        self.model = None

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ffn(x)
        x = self.dropout2(x, training=training)
        x = self.ffn_final(x)
        return x

    def get_config(self):
        config = super(NERModel, self).get_config()
        config.update({"lookup_layer": self.lookup_layer})
        return config

    def train(self, batch_size=32, epochs=2):
        """
        Метод обучает модель для распознавания именнованных сущностей

        Для обучения используются текстовые наборы, размеченные в Label-studio.

        По умолчанию загружается формат CoNLL-2003 (.conll)

        :param batch_size: размер партии (части) данных в 1 итерации обучения
        :param epochs: число итераций (эпох) для обучения
        :return:
        """
        num_train = 1500
        num_val = 338

        """ Загружаем обучающий и проверочный датасет """
        os.makedirs('./ml/data', exist_ok=True)
        raw_train_set = read_conll_2003(filename=f'./ml/data/train_set_{num_train}.conll')
        raw_val_set = read_conll_2003(filename=f'./ml/data/val_set_{num_val}.conll')

        """ Получаем все токены, теги и размерность датасета """
        new_train_set, all_tokens_t, max_len_t = get_tokens_and_tags(dataset=raw_train_set, ner_tags=self.NER_TAGS)
        new_val_set, all_tokens_v, max_len_v = get_tokens_and_tags(dataset=raw_val_set, ner_tags=self.NER_TAGS)

        print(f'MAX_LEN_train: {max_len_t}')
        print(f'MAX_LEN_validation: {max_len_v}')
        if max_len_t > max_len_v:
            MAX_LEN = max_len_t
        else:
            MAX_LEN = max_len_v

        """ Сохраняем в .txt """
        export_to_txt_file(f'./ml/data/conll_train_{num_train}.txt', data=new_train_set)
        export_to_txt_file(f'./ml/data/conll_val_{num_val}.txt', data=new_val_set)

        all_tokens_array = np.array(list(map(str.lower, all_tokens_t)))

        counter = Counter(all_tokens_array)
        print(f'All tokens: {len(counter)}')

        vocab_size = 20000
        """ Получаем и сохраняем словарь """
        # take only (vocab_size - 2) most commons words from the training data since
        # the `StringLookup` class uses 2 additional tokens - one denoting an unknown token
        # and another one denoting a masking token
        vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]
        with open('./ml/data/ner_vocabulary.json', 'w', encoding='utf-8') as json_file:
            json.dump(vocabulary, json_file, ensure_ascii=False)

        # The StringLook class converts tokens to token IDs
        lookup_layer = keras.layers.StringLookup(vocabulary=vocabulary)

        """ Загружаем данные в формате tf.data.Dataset """
        train_data = tf.data.TextLineDataset(f'./ml/data/conll_train_{num_train}.txt')
        val_data = tf.data.TextLineDataset(f'./ml/data/conll_val_{num_val}.txt')

        """ Используем padded_batch ,т.к. каждая запись в датасете имеет разную длину """
        train_dataset = (
            train_data.map(map_record_to_training_data)
            .map(lambda x, y: (lookup_layer(tf.strings.lower(x)), y))
            .padded_batch(batch_size)
        )
        val_dataset = (
            val_data.map(map_record_to_training_data)
            .map(lambda x, y: (lookup_layer(tf.strings.lower(x)), y))
            .padded_batch(batch_size)
        )

        """ Строим и обучаем NER модель """
        ner_model = NERModel(vocab_size=vocab_size, max_len=MAX_LEN, embed_dim=32, num_heads=4, ffn_dim=64,
                             vocabulary=vocabulary)
        ner_model.compile(optimizer='adam',
                          loss=keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy']
                          )
        tb_callback = tf.keras.callbacks.TensorBoard('./ml/data/logs/ner', update_freq='epoch')
        es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=2)
        ner_model.fit(train_dataset, batch_size=batch_size, epochs=epochs, callbacks=[tb_callback, es_callback])

        os.makedirs('./ml/data/models', exist_ok=True)
        ner_model.save(f'./ml/data/models/ner_model_v1_{num_train}.tf', save_format='tf')

        os.makedirs('./ml/data/results', exist_ok=True)
        with open(f'./ml/data/results/model_summary_for_NER.txt', 'w', encoding='utf-8') as file:
            ner_model.summary(print_fn=lambda x: file.write(x + '\n'))

        """ Считаем метрики (precision, recall, F1-score) по всему датасету и сохраняем отчет в файл .txt """
        ner_model.calculate_metrics(val_dataset)

    def calculate_metrics(self, dataset):
        """
        Метод высчитывает метрики без учёта масок
        и печатает отчёт по классификации

        :param dataset: tf.data.Dataset
        :return:
        """
        all_true_tag_ids, all_predicted_tag_ids = [], []

        for x, y in dataset:
            # self = NERModel
            output = self.predict(x)
            predictions = np.argmax(output, axis=-1)
            predictions = np.reshape(predictions, [-1])

            true_tag_ids = np.reshape(y, [-1])

            mask = (true_tag_ids > 0) & (predictions > 0)
            true_tag_ids = true_tag_ids[mask]
            predicted_tag_ids = predictions[mask]

            all_true_tag_ids.append(true_tag_ids)
            all_predicted_tag_ids.append(predicted_tag_ids)

        all_true_tag_ids = np.concatenate(all_true_tag_ids)
        all_predicted_tag_ids = np.concatenate(all_predicted_tag_ids)

        predicted_tags = [self.NER_TAGS[tag] for tag in all_predicted_tag_ids]
        real_tags = [self.NER_TAGS[tag] for tag in all_true_tag_ids]

        os.makedirs('./ml/data/results', exist_ok=True)
        evaluate(real_tags, predicted_tags, file='./ml/data/results/ner_report.txt', verbose=True)

    def load_model(self):
        self.model = keras.models.load_model('./ml/data/models/ner_model_v1_1500.tf')
        with open('./ml/data/ner_vocabulary.json', encoding='utf-8') as json_file:
            self.vocabulary = json.load(json_file)
        self.lookup_layer = keras.layers.StringLookup(vocabulary=self.vocabulary)

    def predicts(self, text: str):
        """
        Метод возвращает распознанные сущности

        :param text: слово, предложение, текст
        :return:
        """
        tokens = text.split()

        sample_input = self.lookup_layer(tf.strings.lower(tokens))
        sample_input = tf.reshape(sample_input, shape=[1, -1])

        output = self.model.predict(sample_input, verbose=2)
        prediction = np.argmax(output, axis=-1)[0]
        prediction = [self.NER_TAGS[i] for i in prediction]
        print(prediction)

        # # Выбираем второй максимум, т.к. модель переобучена на {1: 'O'}
        # prediction_2 = np.argsort(output, axis=-1)[0]
        # prediction_2 = [i[-2] for i in prediction_2]
        # prediction_2 = [self.NER_TAGS[i] for i in prediction_2]
        # print(prediction_2)

    def do_slovnet_ner(self, text: str):
        """
        Альтернативное решение NER с помощью библиотек Slovnet, Navec.

        Метод распознает сущности ОРГАНИЗАЦИЯ, УЧАСТНИК, ГЕОГРАФИЯ

        :param text: слово, предложение, текст
        :return:
        """
        mapping_tags = {'ORG': 'ОРГАНИЗАЦИЯ', 'PER': 'УЧАСТНИК', 'LOC': 'ГЕОГРАФИЯ'}

        markup = self.NER_SLOVNET(text)  # SpanMarkup(text='text', spans=[Span(start=0, stop=12, type='PER|ORG|LOC')])
        result = []
        for span in markup.spans:
            words = markup.text[span.start:span.stop].split(' ')
            # приводим каждое слово к нормальной форме
            sentence = ''
            for word in words:
                sentence += self.MORPH.parse(word)[0].normal_form
                sentence += ' '
            if span.type == 'LOC':
                coordinates = {'lat': 58, 'lon': 39}
                # coordinates = self.GEOCODER.get_coordinates(query=word)
            else:
                coordinates = None
            result.append({'word': sentence,
                           'label': mapping_tags.get(span.type),
                           'coordinates': coordinates})
        return result
