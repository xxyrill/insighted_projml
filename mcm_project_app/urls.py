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
    path('compare/atycb-cas/', views.atycb_cas_comparison_view, name='atycb_cas_comparison'),
    path('compare/atycb-ccis/', views.atycb_ccis_comparison_view, name='atycb_ccis_comparison'),
    path('compare/atycb-cea/', views.atycb_cea_comparison_view, name='atycb_cea_comparison'),
    path('compare/atycb-chs/', views.atycb_chs_comparison_view, name='atycb_chs_comparison'),
    path('compare/atycb-nstp/', views.atycb_nstp_comparison_view, name='atycb_nstp_comparison'),
    path('compare/cas-ccis/', views.cas_ccis_comparison_view, name='cas_ccis_comparison'),
    path('compare/cas-cea/', views.cas_cea_comparison_view, name='cas_cea_comparison'),
    path('compare/cas-chs/', views.cas_chs_comparison_view, name='cas_chs_comparison'),
    path('compare/cas-nstp/', views.cas_nstp_comparison_view, name='cas_nstp_comparison'),
    path('compare/ccis-cea/', views.ccis_cea_comparison_view, name='ccis_cea_comparison'),
    path('compare/ccis-chs/', views.ccis_chs_comparison_view, name='ccis_chs_comparison'),
    path('compare/ccis-nstp/', views.ccis_nstp_comparison_view, name='ccis_nstp_comparison'),
    path('compare/cea-chs/', views.cea_chs_comparison_view, name='cea_chs_comparison'),
    path('compare/cea-nstp/', views.cea_nstp_comparison_view, name='cea_nstp_comparison'),
    path('compare/chs-nstp/', views.chs_nstp_comparison_view, name='chs_nstp_comparison'),
    
    path('comments_table_view/', views.comments_table_view, name='comments_table_view'),

    
    path('api/instructor-ranking-graph/', views.instructor_ranking_graph, name='instructor-ranking-graph'),
    
    path('plot_instructor_ratings/<str:instructor_name>/', views.plot_instructor_ratings, name='plot_instructor_ratings'),
    path('plot_term_1/', views.plot_term_1, name='plot_term_1'),
    path('plot_term_2/', views.plot_term_2, name='plot_term_2'),
    path('plot_all_terms/',views.plot_all_terms, name='plot_all_terms'),
    
 
]
