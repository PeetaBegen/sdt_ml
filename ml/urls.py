from django.urls import path

from . import views

app_name = 'ml'
urlpatterns = [
    path('', views.index, name='index'),
    path('tcl', views.text_classification, name='Text classification'),
    path('tcl/train', views.train_text_classification, name='Text classification training'),
    path('ner', views.named_entity_recognition, name='Named entity recognition'),
    path('ner/train', views.train_named_entity_recognition, name='Named entity recognition training'),
    path('san', views.sentiment_analysis, name='Sentiment analysis'),
    path('san/train', views.train_sentiment_analysis, name='Sentiment analysis training'),
    path('geo', views.geocoding, name='Geocoding'),
    path('predict', views.make_predictions, name='Make predictions'),
]
