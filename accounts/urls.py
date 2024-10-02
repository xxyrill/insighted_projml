from django.urls import path
from .views import register_view, login_view, dashboard_view, dashboardgraph, about_us, logout_view, create_account
from . import views
urlpatterns = [
    path('', login_view, name='login'),

    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('home/', dashboard_view, name='home'),
    path('dashboard/', dashboardgraph, name='dashboard'),
    path('create-account/', create_account, name='create_account'),

    path('upload-csv/', views.upload_csv, name='upload_csv'),
    path('delete_data/', views.delete_data, name='delete_data'),  # Add this line
    path('about/', about_us, name='about'),
]
