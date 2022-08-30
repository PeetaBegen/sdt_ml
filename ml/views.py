from django.http import JsonResponse
from ml.main import MLService


# endpoint: ~/ml
def index(request):
    return JsonResponse({"text": "Hello there! It's ML service"},
                        safe=False, json_dumps_params={'ensure_ascii': False})


# endpoint: ~/ml/predict?file='filename'
def make_predictions(request):
    mlservice = MLService()
    file = request.GET.get(str('file'), './ml/data/NIRMA_comments_1.csv')
    response = mlservice.make_predictions(file=file)
    return JsonResponse(response, safe=False, json_dumps_params={'ensure_ascii': False})


# endpoint: ~/ml/tcl
def text_classification(request):
    mlservice = MLService()
    response = mlservice.do_text_classification()
    return JsonResponse(response, safe=False, json_dumps_params={'ensure_ascii': False})


# endpoint: ~/ml/tcl/train
def train_text_classification(request):
    mlservice = MLService()
    mlservice.train_text_classification()
    response = {"text": "Text classification training is done"}
    return JsonResponse(response, safe=False, json_dumps_params={'ensure_ascii': False})


# endpoint: ~/ml/ner
def named_entity_recognition(request):
    mlservice = MLService()
    mlservice.do_named_entity_recognition()
    response = {"text": "NER is done"}
    return JsonResponse(response, safe=False, json_dumps_params={'ensure_ascii': False})


# endpoint: ~/ml/ner/train
def train_named_entity_recognition(request):
    mlservice = MLService()
    mlservice.train_named_entity_recognition()
    response = {"text": "NER training is done"}
    return JsonResponse(response, safe=False, json_dumps_params={'ensure_ascii': False})


# endpoint: ~/ml/san
def sentiment_analysis(request):
    mlservice = MLService()
    response = mlservice.do_sentiment_analysis()
    return JsonResponse(response, safe=False, json_dumps_params={'ensure_ascii': False})


# endpoint: ~/ml/san/train
def train_sentiment_analysis(request):
    mlservice = MLService()
    mlservice.train_sentiment_analysis()
    response = {"text": "Sentiment analysis training is done"}
    return JsonResponse(response, safe=False, json_dumps_params={'ensure_ascii': False})


# endpoint: ~/ml/geo
def geocoding(request):
    mlservice = MLService()
    response = mlservice.do_geocoding(query='Санкт-Петербург')
    return JsonResponse(response, safe=False, json_dumps_params={'ensure_ascii': False})
