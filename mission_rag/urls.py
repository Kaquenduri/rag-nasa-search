from django.urls import path

from . import views

urlpatterns = [
    path("healthz/", views.health),
    path("api/search/", views.search_sources),
    path("api/research/search/", views.search_research_sources),
    path("api/ask/", views.ask_question),
]
