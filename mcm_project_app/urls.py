from django.urls import path
from . import views

urlpatterns = [
    # path('login/', views.login_page, name='login_page'),
    path('plot-ratings-trend/', views.plot_ratings_trend, name='plot_ratings_trend'),
    path('plot-department-avg-ratings/', views.plot_department_avg_ratings, name='plot_department_avg_ratings'),
    path('plot-rating-distribution/', views.plot_rating_distribution, name='plot_rating_distribution'),
    path('plot-word-clouds/', views.plot_word_clouds, name='plot_word_clouds'),
    path('plot-comment-lengths/', views.plot_comment_lengths, name='plot_comment_lengths'),
    path('plot-ratings-by-course/', views.plot_ratings_by_course, name='plot_ratings_by_course'),
    path('plot-correlation-by-heatmap/', views.plot_correlation_heatmap, name='plot_correlation_heatmap'),
    path('plot-performance-by-year-and-department/', views.plot_performance_by_year_and_department, name='plot_performance_by_year_and_department'),
    path('plot-sentiment-over-time/', views.plot_sentiment_over_time, name='plot_sentiment_over_time'),
    path('plot-instructor-performance-over-time/', views.plot_instructor_performance_over_time, name='plot_instructor_performance_over_time'),
    path('plot-pareto-analysis/', views.plot_pareto_analysis, name='plot_pareto_analysis'),
    path('plot-ratings-vs-comment-length/', views.plot_ratings_vs_comment_length, name='plot_ratings_vs_comment_length'),
    path('plot-instructor-comparison-dashboard/', views.plot_instructor_comparison_dashboard, name='plot_instructor_comparison_dashboard'),
]
