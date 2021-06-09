from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='covid-home'),
    path('simulation/', views.simulation, name='covid-simulation'),
    path('database/', views.database, name='covid-database'),
]
