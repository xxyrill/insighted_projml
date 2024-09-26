from django.urls import path
from . import views

urlpatterns = [
    # path('login/', views.login_page, name='login_page'),
    path('dashboard/generate_graph/', views.generate_graph, name='generate_graph'),
    path('dashboard/', views.dashboard_view, name='dashboard'),  # Use views.dashboard_view
    path('plot-ratings-trend/', views.plot_ratings_trend, name='plot_ratings_trend'),

]
