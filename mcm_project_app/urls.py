from django.urls import path
from . import views

urlpatterns = [
    # path('login/', views.login_page, name='login_page'),
    path('dashboard/generate_graph/', views.generate_graph, name='generate_graph'),
    path('dashboard/', views.dashboard_view, name='dashboard'),  # Use views.dashboard_view
    path('plot_average_ratings_ATYCB/', views.plot_average_ratings_ATYCB, name='plot_average_ratings_ATYCB'),
    path('plot_average_ratings_CAS/', views.plot_average_ratings_CAS, name='plot_average_ratings_CAS'),
    path('plot_average_ratings_CCIS/', views.plot_average_ratings_CCIS, name='plot_average_ratings_CCIS'),
    path('plot_average_ratings_CEA/', views.plot_average_ratings_CEA, name='plot_average_ratings_CEA'),
    path('plot_average_ratings_CHS/', views.plot_average_ratings_CHS, name='plot_average_ratings_CHS'),
    path('plot_average_ratings_NSTP/', views.plot_average_ratings_NSTP, name='plot_average_ratings_NSTP'),
    path('plot_instructor_ratings/<str:instructor_name>/', views.plot_instructor_ratings, name='plot_instructor_ratings'),
    path('plot_comments_pie_chart/', views.plot_comments_pie_chart, name='plot_comments_pie_chart'),
    path('plot_length_of_comments_analysis/', views.plot_length_of_comments_analysis, name='plot_length_of_comments_analysis'),
    #    path('plot_instructor_leaderboard/', views.plot_instructor_leaderboard, name='plot_instructor_leaderboard'),
]
