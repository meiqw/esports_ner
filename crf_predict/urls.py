from django.urls import path
from crf_predict import views

urlpatterns = [path('', views.index, name="homepage"),
               path('linking/<str:entity>/', views.linking, name="linking"),
               path('tagging/<str:doc_title>/', views.tagging, name="tagging")]