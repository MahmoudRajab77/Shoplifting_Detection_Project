from django.urls import path
from .views import IndexView, PredictView, PredictAPIView

app_name = "detector"

urlpatterns = [
    path("",              IndexView.as_view(),      name="index"),
    path("predict/",      PredictView.as_view(),     name="predict"),
    path("api/predict/",  PredictAPIView.as_view(),  name="api_predict"),
]
