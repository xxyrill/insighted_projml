from django.urls import path
from . import views

urlpatterns = [
    # path('login/', views.login_page, name='login_page'),
    path('dashboard/generate_graph/', views.generate_graph, name='generate_graph'),
    path('dashboard/', views.dashboard_view, name='dashboard'),  # Use views.dashboard_view
    path('plot_ratings_trend/', views.plot_ratings_trend, name='plot_ratings_trend'),
    path('plot_department_average_ratings/', views.plot_department_average_ratings, name='plot_department_average_ratings'),
    path('plot_rating_distribution/', views.plot_rating_distribution, name='plot_rating_distribution'),
    path('plot_comments_pie_chart/', views.plot_comments_pie_chart, name='plot_comments_pie_chart'),
#    path('plot_pie_chart_base64/', views.plot_pie_chart_base64, name='plot_pie_chart_base64'),

    path('plot_length_of_comments_analysis/', views.plot_length_of_comments_analysis, name='plot_length_of_comments_analysis'),
    #
# CANNOT BE GRAPH path('plot_ratings_by_course/', views.plot_ratings_by_course, name='plot_ratings_by_course'),
# CANNOT BE GRAPH path('plot_correlation_heatmap/', views.plot_correlation_heatmap, name='plot_correlation_heatmap'),
# CANNOT BE GRAPH path('plot_performance_by_year_and_department/', views.plot_performance_by_year_and_department, name='plot_performance_by_year_and_department'),
    path('plot_sentiment_analysis_over_time/', views.plot_sentiment_analysis_over_time, name='plot_sentiment_analysis_over_time'),
    path('plot_pareto_analysis/', views.plot_pareto_analysis, name='plot_pareto_analysis'),
    path('plot_comparison_of_ratings_and_comment_length/', views.plot_comparison_of_ratings_and_comment_length, name='plot_comparison_of_ratings_and_comment_length'),
# 13
    path('plot_instructor_leaderboard/', views.plot_instructor_leaderboard, name='plot_instructor_leaderboard'),

    # Correlation Heatmap url
]
