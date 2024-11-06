import matplotlib
import plotly.express as px
import plotly.graph_objects as go
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64
import io
import re
import numpy as np
from io import BytesIO
import logging
from django.contrib.auth.decorators import login_required
matplotlib.use('Agg')  # Set the backend to Agg for non-interactive use
from rest_framework.decorators import api_view
from django.db.models import Q
from accounts.models import UploadCSV
from django.db.models.functions import Length, Coalesce
from django.db.models import Avg, Sum, F, Value
from django.db import models
from django.http import HttpResponse
from django.conf import settings
from .utilities import csvPathFileName
from django.shortcuts import render
from .utils import upload_csv_to_db
from django.contrib import messages
from io import BytesIO
from nltk.sentiment import SentimentIntensityAnalyzer # ML sentiment analysis
from django.db import IntegrityError
from nltk.corpus import stopwords
from django.http import JsonResponse, HttpResponseForbidden
from accounts.models import UploadCSV
from django.core.cache import cache
from django.shortcuts import render
from django.views.decorators.http import require_GET
from textblob import TextBlob
from django.apps import apps

def login_page(request):
    return render(request, 'authentications/login.html')

# Delete/Upload database
def upload_csv_to_db(csv_file_path):
    # Print starting message
    print("Starting deletion...")  # Ensure this line is included
    # Step 1: Clear existing data in the UploadCSV table
    try:
        records_deleted = UploadCSV.objects.all().delete()  # This clears all existing records
        print(f"Deleted {records_deleted[0]} records from UploadCSV.")
    except Exception as e:
        print(f"Error while deleting records: {e}")
    # Step 2: Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file_path)  # Load the CSV file
        print("CSV file loaded successfully.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return  # Exit the function if there is an error loading the CSV
    # Step 3: Insert new data into the UploadCSV table
    for _, row in df.iterrows():
        try:
            UploadCSV.objects.create(
                dept_name=row['deptname'],
                resp_fac=row['resp_fac'],
                crs_name=row['crs_name'],
                survey_name=row['survey_name']
            )
        except IntegrityError as e:
            print(f"Error inserting row: {e}")  # Log or handle the error if necessary
    # Step 4: Extract unique values for filters
    departments = df['deptname'].unique()
    instructors = df['resp_fac'].unique()
    courses = df['crs_name'].unique()
    term = df['survey_name'].unique()
    print("Unique Departments:", departments)
    print("Unique Instructors:", instructors)
    print("Unique Courses:", courses)
    print("Unique Survey Name:", term)
    
    return departments, instructors, courses, term
def dashboard_view(request):
    user_role = request.user.user_type  # Assuming user_type stores department or role information
    # Roles with access to all departments
    admin_roles = ['VPAA', 'IDEALS', 'QAEO', 'DEANS', 'HR']
    if user_role in admin_roles:
        # Get all departments and instructors
        departments = list(UploadCSV.objects.values_list('dept_name', flat=True).distinct())
        instructors = list(UploadCSV.objects.values_list('resp_fac', flat=True).distinct())
    else:
        # Filter instructors based on the user's department
        departments = [user_role]
        instructors = list(UploadCSV.objects.filter(dept_name=user_role).values_list('resp_fac', flat=True).distinct())
    courses = list(UploadCSV.objects.values_list('crs_name', flat=True).distinct())
    terms = list(UploadCSV.objects.values_list('survey_name', flat=True).distinct())
    context = {
        'departments': departments,
        'instructors': instructors,
        'courses': courses,
        'terms': terms,
        'user_type': user_role,
        'messages': messages.get_messages(request),  # If you have messages
    }
    return render(request, 'dashboard/dashboard.html', context)
def generate_graph(request):
    # Logic for generating the graph based on request data
    graph_type = request.GET.get('graph', 'default_graph')
    # Example: Call specific graph generation functions
    if graph_type == 'ratings_trend':
        img = plot_ratings_trend(request)
    elif graph_type == 'department_avg_ratings':
        img = plot_department_average_ratings(request)
    # Add logic for other graphs as needed
    # Convert image to base64 for frontend display
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return JsonResponse({'image': img_base64}, safe=False)

def get_unique_filter_options():
    """Function to extract unique filter options for Department, Instructor, and Courses."""
    csv_file_path = csvPathFileName()
    data = pd.read_csv(csv_file_path)
    # Extract unique values for Department, Instructor, and Courses
    departments = data['dept_name'].dropna().unique()  # Assuming 'deptname' is the column name for Department
    instructors = data['resp_fac'].dropna().unique()  # Assuming 'resp_fac' is the column name for Instructor
    courses = data['crs_name'].dropna().unique()  # Assuming 'crs_name' is the column name for Courses
    term = data['survey_name'].dropna().unique()  # Assuming 'crs_name' is the column name for Term
    return departments, instructors, courses
def render_filter_page(request):
    """View to render the filter page with populated dropdowns for Department, Instructor, and Courses."""
    # Get the unique filter options
    departments, instructors, courses, term = get_unique_filter_options()
    # Pass the unique options to the template context
    context = {
        'departments': departments,
        'instructors': instructors,
        'courses': courses,
        'term': term
    }
    return render(request, 'dashboard.html', context)
def get_courses_for_instructor(request):
    instructor = request.GET.get('instructor')
    
    if instructor:
        # Fetch courses for the selected instructor
        courses = list(UploadCSV.objects.filter(resp_fac=instructor).values_list('crs_name', flat=True).distinct())
        return JsonResponse({'courses': courses})
    
    return JsonResponse({'courses': []})


#4. Pie Chart Comments
nltk.download('stopwords')
# Function to get word frequencies

# ATYCB
def plot_average_ratings_ATYCB(request):
    if request.user.user_type not in ['ATYCB', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")

    # Query the database for ATYCB department data
    data = UploadCSV.objects.filter(dept_name='ATYCB').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Extract model fields to access help texts
    model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

    # Create a dictionary for question help texts
    help_texts = {}
    for i in range(1, 31):
        field_name = f'question_{i}'
        help_texts[field_name] = model._meta.get_field(field_name).help_text

    # Calculate average ratings and individual scores
    category_averages = {}
    details = {}

    for category, questions in CATEGORIES.items():
        avg_score = df[questions].mean().mean()  # Overall average for the category
        question_scores = df[questions].mean().tolist()  # Average for each question
        category_averages[category] = avg_score

        # Include help text in details
        details[category] = {help_texts[question]: score for question, score in zip(questions, question_scores)}

    # Prepare data for graph plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

    # Define color based on the average rating
    def get_color(value):
        if value < 1:
            return 'red'
        elif value < 2.49:
            return 'orange'
        elif value < 3.99:
            return 'yellow'
        else:
            return 'green'

    colors = [get_color(avg) for avg in averages]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color=colors)

    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('ATYCB Categorized Average Ratings', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Create a data structure for the hoverable table information
    tables = {}
    for category, scores in details.items():
        tables[category] = [{"Question": question, "Score": f"{score:.2f}"} for question, score in scores.items()]

    # Create a detailed breakdown of the average calculations
    breakdown = {
        category: {
            "Average": f"{avg_score:.2f}",
            "Details": {question: f"{score:.2f}" for question, score in details[category].items()}
        }
        for category, avg_score in category_averages.items()
    }

    return JsonResponse({
        'image': image_base64,
        'tables': tables,
        'breakdown': breakdown
    })

# CAS
def plot_average_ratings_CAS(request):
    if request.user.user_type not in ['CAS', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")

    # Query the database for CAS department data
    data = UploadCSV.objects.filter(dept_name='CAS').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Extract model fields to access help texts
    model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

    # Create a dictionary for question help texts
    help_texts = {}
    for i in range(1, 31):
        field_name = f'question_{i}'
        help_texts[field_name] = model._meta.get_field(field_name).help_text

    # Calculate average ratings and individual scores
    category_averages = {}
    details = {}

    for category, questions in CATEGORIES.items():
        avg_score = df[questions].mean().mean()  # Overall average for the category
        question_scores = df[questions].mean().tolist()  # Average for each question
        category_averages[category] = avg_score

        # Include help text in details
        details[category] = {help_texts[question]: score for question, score in zip(questions, question_scores)}

    # Prepare data for graph plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

    # Define color based on the average rating
    def get_color(value):
        if value < 1:
            return 'red'
        elif value < 2.49:
            return 'orange'
        elif value < 3.99:
            return 'yellow'
        else:
            return 'green'

    colors = [get_color(avg) for avg in averages]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color=colors)

    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CAS Categorized Average Ratings', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Create a data structure for the hoverable table information
    tables = {}
    for category, scores in details.items():
        tables[category] = [{"Question": question, "Score": f"{score:.2f}"} for question, score in scores.items()]

    # Create a detailed breakdown of the average calculations
    breakdown = {
        category: {
            "Average": f"{avg_score:.2f}",
            "Details": {question: f"{score:.2f}" for question, score in details[category].items()}
        }
        for category, avg_score in category_averages.items()
    }

    return JsonResponse({
        'image': image_base64,
        'tables': tables,
        'breakdown': breakdown
    })

# CCIS
def plot_average_ratings_CCIS(request):
    if request.user.user_type not in ['CCIS', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")

    # Query the database for CCIS department data
    data = UploadCSV.objects.filter(dept_name='CCIS').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Extract model fields to access help texts
    model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

    # Create a dictionary for question help texts
    help_texts = {}
    for i in range(1, 31):
        field_name = f'question_{i}'
        help_texts[field_name] = model._meta.get_field(field_name).help_text

    # Calculate average ratings and individual scores
    category_averages = {}
    details = {}

    for category, questions in CATEGORIES.items():
        avg_score = df[questions].mean().mean()  # Overall average for the category
        question_scores = df[questions].mean().tolist()  # Average for each question
        category_averages[category] = avg_score

        # Include help text in details
        details[category] = {help_texts[question]: score for question, score in zip(questions, question_scores)}

    # Prepare data for graph plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

    # Define color based on the average rating
    def get_color(value):
        if value < 1:
            return 'red'
        elif value < 2.49:
            return 'orange'
        elif value < 3.99:
            return 'yellow'
        else:
            return 'green'

    colors = [get_color(avg) for avg in averages]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color=colors)

    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CCIS Categorized Average Ratings', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Create a data structure for the hoverable table information
    tables = {}
    for category, scores in details.items():
        tables[category] = [{"Question": question, "Score": f"{score:.2f}"} for question, score in scores.items()]

    # Create a detailed breakdown of the average calculations
    breakdown = {
        category: {
            "Average": f"{avg_score:.2f}",
            "Details": {question: f"{score:.2f}" for question, score in details[category].items()}
        }
        for category, avg_score in category_averages.items()
    }

    return JsonResponse({
        'image': image_base64,
        'tables': tables,
        'breakdown': breakdown
    })

# CEA
def plot_average_ratings_CEA(request):
    if request.user.user_type not in ['CEA', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")

    # Query the database for CEA department data
    data = UploadCSV.objects.filter(dept_name='CEA').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Extract model fields to access help texts
    model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

    # Create a dictionary for question help texts
    help_texts = {}
    for i in range(1, 31):
        field_name = f'question_{i}'
        help_texts[field_name] = model._meta.get_field(field_name).help_text

    # Calculate average ratings and individual scores
    category_averages = {}
    details = {}

    for category, questions in CATEGORIES.items():
        avg_score = df[questions].mean().mean()  # Overall average for the category
        question_scores = df[questions].mean().tolist()  # Average for each question
        category_averages[category] = avg_score

        # Include help text in details
        details[category] = {help_texts[question]: score for question, score in zip(questions, question_scores)}

    # Prepare data for graph plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

    # Define color based on the average rating
    def get_color(value):
        if value < 1:
            return 'red'
        elif value < 2.49:
            return 'orange'
        elif value < 3.99:
            return 'yellow'
        else:
            return 'green'

    colors = [get_color(avg) for avg in averages]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color=colors)

    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CEA Categorized Average Ratings', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Create a data structure for the hoverable table information
    tables = {}
    for category, scores in details.items():
        tables[category] = [{"Question": question, "Score": f"{score:.2f}"} for question, score in scores.items()]

    # Create a detailed breakdown of the average calculations
    breakdown = {
        category: {
            "Average": f"{avg_score:.2f}",
            "Details": {question: f"{score:.2f}" for question, score in details[category].items()}
        }
        for category, avg_score in category_averages.items()
    }

    return JsonResponse({
        'image': image_base64,
        'tables': tables,
        'breakdown': breakdown
    })

# CHS
def plot_average_ratings_CHS(request):
    if request.user.user_type not in ['CHS', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")

    # Query the database for CHS department data
    data = UploadCSV.objects.filter(dept_name='CHS').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Extract model fields to access help texts
    model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

    # Create a dictionary for question help texts
    help_texts = {}
    for i in range(1, 31):
        field_name = f'question_{i}'
        help_texts[field_name] = model._meta.get_field(field_name).help_text

    # Calculate average ratings and individual scores
    category_averages = {}
    details = {}

    for category, questions in CATEGORIES.items():
        avg_score = df[questions].mean().mean()  # Overall average for the category
        question_scores = df[questions].mean().tolist()  # Average for each question
        category_averages[category] = avg_score

        # Include help text in details
        details[category] = {help_texts[question]: score for question, score in zip(questions, question_scores)}

    # Prepare data for graph plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

    # Define color based on the average rating
    def get_color(value):
        if value < 1:
            return 'red'
        elif value < 2.49:
            return 'orange'
        elif value < 3.99:
            return 'yellow'
        else:
            return 'green'

    colors = [get_color(avg) for avg in averages]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color=colors)

    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CHS Categorized Average Ratings', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Create a data structure for the hoverable table information
    tables = {}
    for category, scores in details.items():
        tables[category] = [{"Question": question, "Score": f"{score:.2f}"} for question, score in scores.items()]

    # Create a detailed breakdown of the average calculations
    breakdown = {
        category: {
            "Average": f"{avg_score:.2f}",
            "Details": {question: f"{score:.2f}" for question, score in details[category].items()}
        }
        for category, avg_score in category_averages.items()
    }

    return JsonResponse({
        'image': image_base64,
        'tables': tables,
        'breakdown': breakdown
    })

# NSTP
def plot_average_ratings_NSTP(request):
    if request.user.user_type not in ['NSTP', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")

    # Query the database for NSTP department data
    data = UploadCSV.objects.filter(dept_name='NSTP').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Extract model fields to access help texts
    model = apps.get_model('accounts', 'UploadCSV')

    # Create a dictionary for question help texts
    help_texts = {}
    for i in range(1, 31):
        field_name = f'question_{i}'
        help_texts[field_name] = model._meta.get_field(field_name).help_text

    # Calculate average ratings and individual scores
    category_averages = {}
    details = {}

    for category, questions in CATEGORIES.items():
        avg_score = df[questions].mean().mean()  # Overall average for the category
        question_scores = df[questions].mean().tolist()  # Average for each question
        category_averages[category] = avg_score

        # Include help text in details
        details[category] = {help_texts[question]: score for question, score in zip(questions, question_scores)}

    # Prepare the breakdown
    breakdown = {
        category: {
            "Average": f"{category_averages[category]:.2f}",
            "Details": {question: f"{score:.2f}" for question, score in details[category].items()}
        }
        for category in category_averages
    }

    # Return the breakdown in JSON format
    return JsonResponse({
        'breakdown': breakdown
    })
    
# DEPARTMENT COMPARISON GRAPHS
# ATYCB & CAS
def atycb_cas_comparison_view(request):
    if request.user.user_type not in ['ATYCB', 'CAS', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not ATYCB & CAS user

    departments = ['ATYCB', 'CAS']  # Both ATYCB and CAS
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['ATYCB'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "ATYCB_Average": f"{department_averages['ATYCB'][category]:.2f}",
            "CAS_Average": f"{department_averages['CAS'][category]:.2f}",
            "Details": {
                "ATYCB": {question: f"{score:.2f}" for question, score in details['ATYCB'][category].items()},
                "CAS": {question: f"{score:.2f}" for question, score in details['CAS'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# ATYCB & CCIS
def atycb_ccis_comparison_view(request):
    if request.user.user_type not in ['ATYCB', 'CCIS', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not ATYCB & CCIS user

    departments = ['ATYCB', 'CCIS']  # Both ATYCB and CCIS
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['ATYCB'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "ATYCB_Average": f"{department_averages['ATYCB'][category]:.2f}",
            "CCIS_Average": f"{department_averages['CCIS'][category]:.2f}",
            "Details": {
                "ATYCB": {question: f"{score:.2f}" for question, score in details['ATYCB'][category].items()},
                "CCIS": {question: f"{score:.2f}" for question, score in details['CCIS'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })


# ATYCB & CEA 
def atycb_cea_comparison_view(request):
    if request.user.user_type not in ['ATYCB', 'CEA', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not ATYCB & CEA user

    departments = ['ATYCB', 'CEA']  # Both ATYCB and CEA
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['ATYCB'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "ATYCB_Average": f"{department_averages['ATYCB'][category]:.2f}",
            "CEA_Average": f"{department_averages['CEA'][category]:.2f}",
            "Details": {
                "ATYCB": {question: f"{score:.2f}" for question, score in details['ATYCB'][category].items()},
                "CEA": {question: f"{score:.2f}" for question, score in details['CEA'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })


# ATYCB & CHS
def atycb_chs_comparison_view(request):
    if request.user.user_type not in ['ATYCB', 'CHS', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not ATYCB & CHS user

    departments = ['ATYCB', 'CHS']  # Both ATYCB and CHS
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['ATYCB'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "ATYCB_Average": f"{department_averages['ATYCB'][category]:.2f}",
            "CHS_Average": f"{department_averages['CHS'][category]:.2f}",
            "Details": {
                "ATYCB": {question: f"{score:.2f}" for question, score in details['ATYCB'][category].items()},
                "CHS": {question: f"{score:.2f}" for question, score in details['CHS'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# ATYCB & NSTP
def atycb_nstp_comparison_view(request):
    if request.user.user_type not in ['ATYCB', 'NSTP', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not ATYCB & NSTP user

    departments = ['ATYCB', 'NSTP']  # Both ATYCB and NSTP
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['ATYCB'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "ATYCB_Average": f"{department_averages['ATYCB'][category]:.2f}",
            "NSTP_Average": f"{department_averages['NSTP'][category]:.2f}",
            "Details": {
                "ATYCB": {question: f"{score:.2f}" for question, score in details['ATYCB'][category].items()},
                "NSTP": {question: f"{score:.2f}" for question, score in details['NSTP'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# CAS & CCIS GRAPH
def cas_ccis_comparison_view(request):
    if request.user.user_type not in ['CAS', 'CCIS', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CAS & CCIS user

    departments = ['CAS', 'CCIS']  # Both CAS and CCIS
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['CAS'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "CAS_Average": f"{department_averages['CAS'][category]:.2f}",
            "CCIS_Average": f"{department_averages['CCIS'][category]:.2f}",
            "Details": {
                "CAS": {question: f"{score:.2f}" for question, score in details['CAS'][category].items()},
                "CCIS": {question: f"{score:.2f}" for question, score in details['CCIS'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# CAS & CEA
def cas_cea_comparison_view(request):
    if request.user.user_type not in ['CAS', 'CEA', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CAS & CEA user

    departments = ['CAS', 'CEA']  # Both CAS and CEA
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['CAS'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "CAS_Average": f"{department_averages['CAS'][category]:.2f}",
            "CEA_Average": f"{department_averages['CEA'][category]:.2f}",
            "Details": {
                "CAS": {question: f"{score:.2f}" for question, score in details['CAS'][category].items()},
                "CEA": {question: f"{score:.2f}" for question, score in details['CEA'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# CAS & CHS
def cas_chs_comparison_view(request):
    if request.user.user_type not in ['CAS', 'CHS', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CAS & CHS user

    departments = ['CAS', 'CHS']  # Both CAS and CHS
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['CAS'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "CAS_Average": f"{department_averages['CAS'][category]:.2f}",
            "CHS_Average": f"{department_averages['CHS'][category]:.2f}",
            "Details": {
                "CAS": {question: f"{score:.2f}" for question, score in details['CAS'][category].items()},
                "CHS": {question: f"{score:.2f}" for question, score in details['CHS'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# CAS & NSTP
def cas_nstp_comparison_view(request):
    if request.user.user_type not in ['CAS', 'NSTP', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CAS & NSTP user

    departments = ['CAS', 'NSTP']  # Both CAS and NSTP
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['CAS'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "CAS_Average": f"{department_averages['CAS'][category]:.2f}",
            "NSTP_Average": f"{department_averages['NSTP'][category]:.2f}",
            "Details": {
                "CAS": {question: f"{score:.2f}" for question, score in details['CAS'][category].items()},
                "NSTP": {question: f"{score:.2f}" for question, score in details['NSTP'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# CCIS & CEA
def ccis_cea_comparison_view(request):
    if request.user.user_type not in ['CCIS', 'CEA', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CCIS & CEA user

    departments = ['CCIS', 'CEA']  # Both CCIS and CEA
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['CCIS'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "CCIS_Average": f"{department_averages['CCIS'][category]:.2f}",
            "CEA_Average": f"{department_averages['CEA'][category]:.2f}",
            "Details": {
                "CCIS": {question: f"{score:.2f}" for question, score in details['CCIS'][category].items()},
                "CEA": {question: f"{score:.2f}" for question, score in details['CEA'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# CCIS & CHS
def ccis_chs_comparison_view(request):
    if request.user.user_type not in ['CCIS', 'CHS', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CCIS & CHS user

    departments = ['CCIS', 'CHS']  # Both CCIS and CHS
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['CCIS'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "CCIS_Average": f"{department_averages['CCIS'][category]:.2f}",
            "CHS_Average": f"{department_averages['CHS'][category]:.2f}",
            "Details": {
                "CCIS": {question: f"{score:.2f}" for question, score in details['CCIS'][category].items()},
                "CHS": {question: f"{score:.2f}" for question, score in details['CHS'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# CCIS & NSTP
def ccis_nstp_comparison_view(request):
    if request.user.user_type not in ['CCIS', 'NSTP', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CCIS & NSTP user

    departments = ['CCIS', 'NSTP']  # Both CCIS and NSTP
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['CCIS'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "CCIS_Average": f"{department_averages['CCIS'][category]:.2f}",
            "NSTP_Average": f"{department_averages['NSTP'][category]:.2f}",
            "Details": {
                "CCIS": {question: f"{score:.2f}" for question, score in details['CCIS'][category].items()},
                "NSTP": {question: f"{score:.2f}" for question, score in details['NSTP'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# CEA & CHS
def cea_chs_comparison_view(request):
    if request.user.user_type not in ['CEA', 'CHS', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CEA & CHS user

    departments = ['CEA', 'CHS']  # Both CEA and CHS
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['CEA'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "CEA_Average": f"{department_averages['CEA'][category]:.2f}",
            "CHS_Average": f"{department_averages['CHS'][category]:.2f}",
            "Details": {
                "CEA": {question: f"{score:.2f}" for question, score in details['CEA'][category].items()},
                "CHS": {question: f"{score:.2f}" for question, score in details['CHS'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# CEA & NSTP
def cea_nstp_comparison_view(request):
    if request.user.user_type not in ['CEA', 'NSTP', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CEA & NSTP user

    departments = ['CEA', 'NSTP']  # Both CEA and NSTP
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['CEA'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "CEA_Average": f"{department_averages['CEA'][category]:.2f}",
            "NSTP_Average": f"{department_averages['NSTP'][category]:.2f}",
            "Details": {
                "CEA": {question: f"{score:.2f}" for question, score in details['CEA'][category].items()},
                "NSTP": {question: f"{score:.2f}" for question, score in details['NSTP'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# CHS & NSTP
def chs_nstp_comparison_view(request):
    if request.user.user_type not in ['CHS', 'NSTP', 'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CHS & NSTP user

    departments = ['CHS', 'NSTP']  # Both CHS and NSTP
    department_averages = {}
    details = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Extract model fields to access help texts
        model = apps.get_model('accounts', 'UploadCSV')  # Replace 'your_app_name' with your actual app name

        # Create a dictionary for question help texts
        help_texts = {}
        for i in range(1, 31):
            field_name = f'question_{i}'
            help_texts[field_name] = model._meta.get_field(field_name).help_text

        # Define categories and corresponding question columns
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge & Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {}
        dept_details = {}

        for category, questions in CATEGORIES.items():
            avg_score = df[questions].mean().mean()  # Overall average for the category
            if pd.isnull(avg_score):
                avg_score = 0  # Default to 0 if no valid data

            question_scores = df[questions].mean().fillna(0).tolist()  # Average for each question
            category_averages[category] = avg_score

            # Include help text in details for hoverable information
            dept_details[category] = {
                help_texts[question]: score for question, score in zip(questions, question_scores)
            }

        department_averages[dept] = category_averages
        details[dept] = dept_details

    # Prepare the breakdown structure for the JavaScript
    combined_breakdown = {}
    categories = list(department_averages['CHS'].keys())

    for category in categories:
        combined_breakdown[category] = {
            "CHS_Average": f"{department_averages['CHS'][category]:.2f}",
            "NSTP_Average": f"{department_averages['NSTP'][category]:.2f}",
            "Details": {
                "CHS": {question: f"{score:.2f}" for question, score in details['CHS'][category].items()},
                "NSTP": {question: f"{score:.2f}" for question, score in details['NSTP'][category].items()}
            }
        }

    # Create a response with the breakdown
    return JsonResponse({
        'breakdown': combined_breakdown
    })

# INSTRUCTOR AVERAGE RATINGS
def get_color(value):
    if value < 1:
        return 'red'
    elif value < 2.49:
        return 'orange'
    elif value < 3.99:
        return 'yellow'
    else:
        return 'green'

CATEGORIES = {
    "Presence/Guidance": [f"question_{i}" for i in range(1, 4)],
    "Collaborative Learning": [f"question_{i}" for i in range(4, 10)],
    "Active Learning": [f"question_{i}" for i in range(10, 13)],
    "Content Knowledge and Proficiency": [f"question_{i}" for i in range(13, 16)],
    "Course Expectations": [f"question_{i}" for i in range(16, 19)],
    "Clarity/Instructions": [f"question_{i}" for i in range(19, 22)],
    "Feedback": [f"question_{i}" for i in range(22, 25)],
    "Inclusivity": [f"question_{i}" for i in range(25, 28)],
    "Outcome-Based Education": [f"question_{i}" for i in range(28, 31)],
}

def plot_instructor_ratings(request, instructor_name):
    # Check if the user has permission to access this data
    if request.user.user_type not in ['ATYCB','CCIS', 'CAS', 'CEA',  'VPAA', 'IDEALS', 'QAEO', 'HR']:
        return HttpResponseForbidden("You do not have permission to access this data.")

    # Query the database for the selected instructor
    data = UploadCSV.objects.filter(resp_fac=instructor_name).values(
        *[f"question_{i}" for i in range(1, 31)]
    )
    df = pd.DataFrame(data)

    # Convert question columns to numeric and calculate category averages
    df = df.apply(pd.to_numeric, errors="coerce")

    # Extract model fields to access help texts
    model = apps.get_model('accounts', 'UploadCSV')

    # Create a dictionary for question help texts
    help_texts = {}
    for i in range(1, 31):
        field_name = f'question_{i}'
        help_texts[field_name] = model._meta.get_field(field_name).help_text

    # Calculate average ratings and individual scores
    category_averages = {}
    details = {}
    category_colors = {}

    for category, questions in CATEGORIES.items():
        avg_score = df[questions].mean().mean()  # Overall average for the category
        question_scores = df[questions].mean().tolist()  # Average for each question
        category_averages[category] = avg_score

        # Include help text in details
        details[category] = {help_texts[question]: score for question, score in zip(questions, question_scores)}
        
        # Assign a color based on the average score using the defined color scale
        category_colors[category] = get_color(avg_score)

    # Prepare the breakdown with color information
    breakdown = {
        category: {
            "Average": f"{category_averages[category]:.2f}",
            "Color": category_colors[category],  # Add color information here
            "Details": {question: f"{score:.2f}" for question, score in details[category].items()}
        }
        for category in category_averages
    }

    # Return the breakdown in JSON format
    return JsonResponse({
        'breakdown': breakdown
    })

# COURSE GRAPH
def plot_average_ratings_by_course(request):
    course = request.GET.get('course')

    if not course:
        return JsonResponse({'error': 'Course must be selected.'}, status=400)

    print(f"Received course: {course}")  # Debugging: Print received course name

    # Query the database for data for the selected course
    data = UploadCSV.objects.filter(crs_name=course).values(
        'crs_name', *['question_{}'.format(i) for i in range(1, 31)]
    )
    df = pd.DataFrame(list(data))

    if df.empty:
        return JsonResponse({'error': 'No data available for the selected course.'}, status=400)

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    model = apps.get_model('accounts', 'UploadCSV')

    # Create a dictionary for question help texts
    help_texts = {f'question_{i}': model._meta.get_field(f'question_{i}').help_text for i in range(1, 31)}

    # Calculate average ratings and individual scores
    category_averages = {}
    details = {}

    # Assuming you have a CATEGORIES dictionary mapping categories to their questions
    for category, questions in CATEGORIES.items():
        avg_score = df[questions].mean().mean()  # Overall average for the category
        question_scores = df[questions].mean().tolist()  # Average for each question
        category_averages[category] = avg_score

        # Include help text in details with color based on the score
        details[category] = {
            help_texts[question]: {
                'Score': f"{score:.2f}",
                'Color': get_color(score)
            }
            for question, score in zip(questions, question_scores)
        }

    # Prepare the JSON response
    response_data = {
        "breakdown": {
            category: {
                "Average": f"{category_averages[category]:.2f}",
                "Color": "#9ACD32",  # Or calculate based on some logic
                "Details": details[category]
            }
            for category in CATEGORIES
        }
    }

    return JsonResponse(response_data)

# TERM TREND GRAPH
def get_terms_data():
    # Query the database for data from all terms, selecting the term and question fields
    terms_data = UploadCSV.objects.values('term', *['question_{}'.format(i) for i in range(1, 31)])

    if not terms_data.exists():
        return None

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(list(terms_data))

    # Convert question columns to numeric to handle possible NaN values
    question_columns = ['question_{}'.format(i) for i in range(1, 31)]
    df[question_columns] = df[question_columns].apply(pd.to_numeric, errors='coerce')

    return df

def get_color(value):
    if value < 1:
        return 'red'
    elif value < 2.49:
        return 'orange'
    elif value < 3.99:
        return 'yellow'
    else:
        return 'green'

def plot_term_trend(term_filter):
    df = get_terms_data()

    if df is None:
        return JsonResponse({'error': 'No data available'}, status=400)

    # Strip leading/trailing spaces from both term filter and term column values
    df['term'] = df['term'].astype(str).str.strip()  # Remove whitespace from term values
    filtered_df = df[df['term'] == term_filter.strip()]  # Remove whitespace from the filter

    if filtered_df.empty:
        return JsonResponse({'error': f'Term \"{term_filter}\" not available'}, status=400)

    # Calculate the overall score as the average of all question_* fields
    filtered_df['overall_score'] = filtered_df.iloc[:, 1:].mean(axis=1)

    # Calculate average overall scores per term
    term_average = filtered_df['overall_score'].mean()

    # Prepare data for graph plotting
    terms = [term_filter]  # List of terms for the graph
    averages = [term_average]  # Average for the selected term

    # Determine the color for the bar based on the average score using the get_color function
    colors = [get_color(term_average)]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(terms, averages, color=colors, label='Average Rating')

    # Add value labels on top of each bar
    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Terms', fontsize=14)
    plt.ylabel('Average Overall Score', fontsize=14)
    plt.title(f'Trend of Overall Score for {term_filter}', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)
    plt.legend()

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return the base64-encoded image in JSON response
    return JsonResponse({'image': image_base64})

def plot_term_1(request):
    return plot_term_trend('Term 1')

def plot_term_2(request):
    return plot_term_trend('Term 2')

def plot_term_3(request):
    return plot_term_trend('Term 3')

def get_color(value):
    if value < 1:
        return 'red'
    elif value < 2.49:
        return 'orange'
    elif value < 3.99:
        return 'yellow'
    else:
        return 'green'

def plot_all_terms(request):
    df = get_terms_data()

    if df is None:
        return JsonResponse({'error': 'No data available'}, status=400)

    # Calculate the overall score as the average of all question_* fields
    df['overall_score'] = df.iloc[:, 1:].mean(axis=1)

    # Calculate average overall scores per term
    term_averages = df.groupby('term')['overall_score'].mean().reset_index()

    if term_averages.empty:
        return JsonResponse({'error': 'No terms data found.'}, status=400)

    # Prepare data for graph plotting
    terms = term_averages['term'].tolist()  # e.g. ['Term 1', 'Term 2']
    averages = term_averages['overall_score'].tolist()

    # Determine the color for each term based on its average score using the get_color function
    colors = [get_color(avg) for avg in averages]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(terms, averages, color=colors, label='Average Rating')

    # Add value labels on top of each bar
    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add a connecting line between the bars
    plt.plot(terms, averages, marker='o', color='red', label='Trend Line', linestyle='-')

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Terms', fontsize=14)
    plt.ylabel('Average Overall Score', fontsize=14)
    plt.title('Trend of Overall Scores for All Terms', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)
    plt.legend()

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return the base64-encoded image in JSON response
    return JsonResponse({'image': image_base64})

# COMMENT ANALYSIS GRAPH
# GET INSTRUCTORS BY TERM
def get_instructors_by_term(request):
    term = request.GET.get('term', None)
    if term:
        instructors = UploadCSV.objects.filter(term=term).values_list('resp_fac', flat=True).distinct()
        return JsonResponse({'instructors': list(instructors)})
    return JsonResponse({'instructors': []})

# GET COURSES BY INSTRUCTOR
def get_courses_by_instructor(request):
    term = request.GET.get('term', None)
    instructor = request.GET.get('instructor', None)
    if term and instructor:
        courses = UploadCSV.objects.filter(term=term, resp_fac=instructor).values_list('crs_name', flat=True).distinct()
        return JsonResponse({'courses': list(courses)})
    return JsonResponse({'courses': []})

@login_required
def comments_table_view(request):
    term_filter = request.GET.get('term', 'all')
    instructor_filter = request.GET.get('instructor', '')
    course_filter = request.GET.get('course', '')
    comment_type_filter = request.GET.get('comment_type', 'both')
    sentiment_filter = request.GET.get('sentiment', 'all')

    # Log the filters for debugging
    print(f"Term Filter: {term_filter}, Instructor Filter: {instructor_filter}, Course Filter: {course_filter}, Comment Type Filter: {comment_type_filter}, Sentiment Filter: {sentiment_filter}")

    # Get user role for access control
    user_role = request.user.user_type  # Assuming 'user_type' is a field in your User model

    # Define allowed departments for specific user roles
    role_department_map = {
        'ATYCB': 'ATYCB',
        'CAS': 'CAS',
        'CEA': 'CEA',
        'CCIS': 'CCIS'
    }

    # Restrict access based on user role
    if user_role in role_department_map:
        allowed_department = role_department_map[user_role]

        # Ensure the user is only accessing their department's data
        if instructor_filter and not UploadCSV.objects.filter(resp_fac=instructor_filter, dept_name=allowed_department).exists():
            return JsonResponse({'error': 'Not authorized: Instructor is not in your department.'}, status=403)

        if course_filter and not UploadCSV.objects.filter(crs_name=course_filter, dept_name=allowed_department).exists():
            return JsonResponse({'error': 'Not authorized: Course is not in your department'}, status=403)

    # Base query: filter by term if selected
    comments = UploadCSV.objects.all()

    if term_filter and term_filter != 'all':
        comments = comments.filter(term=term_filter)
    if instructor_filter:
        comments = comments.filter(resp_fac=instructor_filter)
    if course_filter:
        comments = comments.filter(crs_name=course_filter)

    # Select relevant fields
    comments = comments.values('question_31', 'question_32', 'term')

    # Exclusion pattern to remove unwanted comments
    exclusion_pattern = r'^(None|none|N/A|n/a|\.{1,2}|[-.]*|okay|ok|Meh|Wala|wala|hm| ,)$'

    # Filter comments by type and length
    if comment_type_filter == 'instructor':
        comments = comments.exclude(Q(question_31__regex=exclusion_pattern) | Q(question_31__isnull=True) | Q(question_31__exact='')).annotate(
            comment_length=Length('question_31')
        ).filter(comment_length__gte=4)
    elif comment_type_filter == 'course':
        comments = comments.exclude(Q(question_32__regex=exclusion_pattern) | Q(question_32__isnull=True) | Q(question_32__exact='')).annotate(
            comment_length=Length('question_32')
        ).filter(comment_length__gte=4)
    else:
        comments = comments.annotate(
            instructor_length=Length('question_31'),
            course_length=Length('question_32')
        ).exclude(
            Q(question_31__regex=exclusion_pattern) | Q(question_31__isnull=True) | Q(question_31__exact='') |
            Q(question_32__regex=exclusion_pattern) | Q(question_32__isnull=True) | Q(question_32__exact='')
        ).filter(instructor_length__gte=4, course_length__gte=4)

    # Classify comments using TextBlob for sentiment analysis
    for comment in comments:
        comment['sentiment'] = 'neutral'  # Default sentiment

        if comment_type_filter in ['instructor', 'both'] and comment.get('question_31'):
            instructor_comment = comment['question_31']
            analysis = TextBlob(instructor_comment)
            polarity = analysis.sentiment.polarity

            if polarity > 0.2:  # Positive threshold
                comment['sentiment'] = 'good'
            elif polarity < -0.2:  # Negative threshold
                comment['sentiment'] = 'bad'

        if comment_type_filter in ['course', 'both'] and comment.get('question_32'):
            course_comment = comment['question_32']
            analysis = TextBlob(course_comment)
            polarity = analysis.sentiment.polarity

            if polarity > 0.2:  # Positive threshold
                comment['sentiment'] = 'good'
            elif polarity < -0.2:  # Negative threshold
                comment['sentiment'] = 'bad'

    # Filter comments based on sentiment if needed
    if sentiment_filter == 'good':
        comments = [comment for comment in comments if comment['sentiment'] == 'good']
    elif sentiment_filter == 'bad':
        comments = [comment for comment in comments if comment['sentiment'] == 'bad']
    elif sentiment_filter == 'neutral':
        comments = [comment for comment in comments if comment['sentiment'] == 'neutral']

    # Check if there are no comments after filtering
    if not comments:
        return JsonResponse({'error': 'No data available for the selected filters.'}, status=404)

    print(f"Filtered Comments: {list(comments)}")
    return JsonResponse({
        'comments': list(comments)
    })

# INSTRUCTOR RANKING
def generate_instructor_graph(department_selected=None, term_selected=None):
    # Start with all instructors
    instructors = UploadCSV.objects.all()

    # Filter by department if provided
    if department_selected and department_selected != 'all':
        instructors = instructors.filter(dept_name=department_selected)
        print(f"Filtered by department: {department_selected} - Number of instructors: {instructors.count()}")

    # Filter by term if provided and valid
    if term_selected and term_selected.lower() not in ['all', 'all terms']:
        instructors = instructors.filter(term=term_selected)
        print(f"Filtered by term: {term_selected} - Number of instructors: {instructors.count()}")

    if not instructors.exists():
        print("No instructors found for the selected filters.")
        return JsonResponse({'message': 'No data available for the selected term and department.'}, status=404)

    # Initialize dictionary to hold scores
    average_scores = {}

    for instructor in instructors:
        instructor_name = instructor.resp_fac
        total_score = 0
        question_count = 30  # Number of questions to consider

        # Calculate total score from question_1 to question_30
        for i in range(1, question_count + 1):
            score = getattr(instructor, f'question_{i}', None)
            if score is None:
                continue
            total_score += score

        # Update the average score for the instructor
        if instructor_name not in average_scores:
            average_scores[instructor_name] = []

        average_scores[instructor_name].append(total_score / question_count)

    # Check if there are any instructors with valid scores
    if not average_scores:
        print("No valid scores found for any instructors.")
        return None

    # Prepare data for plotting and calculate averages
    names = list(average_scores.keys())
    averages = [sum(scores) / len(scores) for scores in average_scores.values()]

    # Combine names and averages and sort by averages
    leaderboard = list(zip(names, averages))
    leaderboard.sort(key=lambda x: x[1], reverse=True)

    # Limit to top 10 instructors
    top_leaderboard = leaderboard[:10]
    top_names = [name for name, _ in top_leaderboard]
    top_averages = [average for _, average in top_leaderboard]

    # Reverse the order of names and averages for highest at the top
    top_names.reverse()
    top_averages.reverse()

    # Determine colors for the bars
    bar_colors = [get_color(avg) for avg in top_averages]

    # Plot the leaderboard
    plt.figure(figsize=(16, 9))
    bars = plt.barh(top_names, top_averages, color=bar_colors)

    # Set titles and labels with enhanced visuals
    plt.title(f'Top 10 Instructor Leaderboard\nTerm: {term_selected if term_selected else "All Terms"} | Department: {department_selected if department_selected else "All Departments"}',
              fontsize=15, fontweight='bold')
    plt.xlabel('Average Rating', fontsize=10, fontweight='bold')
    plt.ylabel('Instructors', fontsize=10, fontweight='bold')

    # Customize grid lines and annotations
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    for bar in bars:
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.2f}', va='center', ha='left', fontsize=12, color='black')

    # Improve overall visual style
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)  # Higher DPI for better quality
    plt.close()
    buf.seek(0)

    # Encode to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@require_GET
def instructor_ranking_graph(request):
    department_selected = request.GET.get('dept_name', None)
    term_selected = request.GET.get('term_selected', None)

    # Check if the user is logged in and has a specific role
    if request.user.is_authenticated:
        user_role = request.user.user_type  # Assuming 'user_type' is a field in the user model

        # Define allowed departments for specific user roles
        role_department_map = {
            'ATYCB': 'ATYCB',
            'CAS': 'CAS',
            'CEA': 'CEA',
            'CCIS': 'CCIS'
        }

        # Restrict access based on user role
        if user_role in role_department_map:
            allowed_department = role_department_map[user_role]

            # Check if the user is trying to access an allowed department
            if department_selected != allowed_department:
                return JsonResponse({'error': f'Access denied. You can only view graphs for your department: {allowed_department}.'}, status=403)

    # Add logging for debugging purposes
    print(f'Request received. Department Selected: {department_selected}, Term Selected: {term_selected}')

    # Generate the graph
    graph_image = generate_instructor_graph(department_selected, term_selected)

    # Handle case where no graph is generated
    if graph_image is None:
        print("Graph could not be generated, no data available.")
        return JsonResponse({'error': 'No data found for the selected filters'}, status=400)

    print("Graph generated successfully.")
    return JsonResponse({'graph': graph_image})