from django.urls import path

from . import views

urlpatterns = [
    path("healthz/", views.health),
    path("api/search/", views.search_sources),
    path("api/ask/", views.ask_question),
]
