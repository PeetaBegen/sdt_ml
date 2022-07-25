from django.http import JsonResponse
from ml.main import MLService


def index(request):
    return JsonResponse({"text": "Hello there!"})


def make_predictions(request):
    mlservice = MLService()
    response = mlservice.make_predictions()
    return JsonResponse(response)


def text_classification(request):
    mlservice = MLService()
    response = mlservice.do_text_classification()
    return JsonResponse(response)


def train_text_classification(request):
    mlservice = MLService()
    mlservice.train_text_classification()
    response = {"text": "Text classification training is done"}
    return JsonResponse(response)


def named_entity_recognition(request):
    mlservice = MLService()
    mlservice.do_named_entity_recognition()
    response = {"text": "NER done"}
    return JsonResponse(response)


def train_named_entity_recognition(request):
    mlservice = MLService()
    mlservice.train_named_entity_recognition()
    response = {"text": "NER training is done"}
    return JsonResponse(response)


def sentiment_analysis(request):
    mlservice = MLService()
    response = mlservice.do_sentiment_analysis()
    return JsonResponse(response)


def train_sentiment_analysis(request):
    mlservice = MLService()
    mlservice.train_sentiment_analysis()
    response = {"text": "Sentiment analysis training is done"}
    return JsonResponse(response)


def geocoding(request):
    mlservice = MLService()
    response = mlservice.do_geocoding(query='Санкт-Петербург')
    return JsonResponse(response)
