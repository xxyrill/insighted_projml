from django.urls import path
from .views import register_view, login_view, dashboard_view, dashboardgraph, logout_view
from . import views
urlpatterns = [
    path('', login_view, name='login'),

    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('home/', dashboard_view, name='home'),
    path('dashboard/', dashboardgraph, name='dashboard'),

    path('upload-csv/', views.upload_csv, name='upload_csv'),
]
