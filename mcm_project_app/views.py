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

matplotlib.use('Agg')  # Set the backend to Agg for non-interactive use
from rest_framework.decorators import api_view
from django.db.models.functions import Length
from django.db.models import Q
from transformers import pipeline
from django.db.models import Avg, Sum, F
from django.db import models
from django.http import HttpResponse
from django.conf import settings
from .utilities import csvPathFileName
from django.shortcuts import render
from .utils import upload_csv_to_db
from django.contrib import messages
from io import BytesIO
from nltk.sentiment import SentimentIntensityAnalyzer # machine learning sentiment analysis
from django.db import IntegrityError
from nltk.corpus import stopwords
from django.http import JsonResponse, HttpResponseForbidden
from accounts.models import UploadCSV
from django.core.cache import cache
from django.shortcuts import render
from django.views.decorators.http import require_GET
from textblob import TextBlob

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
    departments = list(UploadCSV.objects.values_list('dept_name', flat=True).distinct())  # Convert QuerySet to list
    instructors = list(UploadCSV.objects.values_list('resp_fac', flat=True).distinct())  # Convert QuerySet to list
    courses = list(UploadCSV.objects.values_list('crs_name', flat=True).distinct())  # Convert QuerySet to list
    terms = list(UploadCSV.objects.values_list('survey_name', flat=True).distinct())  # Convert QuerySet to list
    
    context = {
        'departments': departments,
        'instructors': instructors,
        'courses': courses,
        'terms': terms,
        'user_type': request.user.user_type,
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


#4. Pie Chart Comments
nltk.download('stopwords')
# Function to get word frequencies

### DEPARTMENT AVERAGE RATINGS
def plot_average_ratings_ATYCB(request):
    if request.user.user_type != 'ATYCB':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not ATYCB user

    
    # Query the database for ATYCB department data
    data = UploadCSV.objects.filter(dept_name='ATYCB').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

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

    # Calculate average ratings
    category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}

    # Prepare data for graph plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

    # Define color based on the average rating
    def get_color(value):
        if value < 1:  # Below threshold for red
            return 'red'
        elif value < 2: 
            return 'orange'
        elif value < 3:  
            return 'yellow'
        elif value < 4: 
            return '#9ACD32'                
        else:  # Above threshold for green
            return 'green'

    # Generate colors for each bar based on average ratings
    colors = [get_color(avg) for avg in averages]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color=colors)

    # Add value labels on top of each bar
    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('ATYCB Categorized Average Ratings', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Create a data structure for the hoverable table information
    tables = {
        "Presence/Guidance": [
            {"Indicator": "The instructor has set clear standards regarding their timeliness in responding to messages"},
            {"Indicator": "The instructor has provided the appropriate information and contact details for technical concern"},
            {"Indicator": "The instructor showed interest in student progress"}
        ],
        "Collaborative Learning": [
            {"Indicator": "The instructor encourages learners to participate"},
            {"Indicator": "The instructor implements Small Group Discussions (Breakout Rooms)"},
            {"Indicator": "The instructor provides equal opportunities for students to share ideas and viewpoints"}
        ],
        # Add other categories similarly...
    }

    # Return JSON response with image and table data
    return JsonResponse({
        'image': image_base64,
        'tables': tables
    })

def plot_average_ratings_CAS(request):
    if request.user.user_type != 'CAS':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CAS user
        
    # Query the database for CAS department data
    data = UploadCSV.objects.filter(dept_name='CAS').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

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

    # Calculate average ratings
    category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}

    # Prepare data for graph plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())
    
    # Define color based on the average rating
    def get_color(value):
        if value < 1:  # Below threshold for red
            return 'red'
        elif value < 2: 
            return 'orange'
        elif value < 3:  
            return 'yellow'
        elif value < 4: 
            return '#9ACD32'                
        else:  # Above threshold for green
            return 'green'
        
    # Generate colors for each bar based on average ratings
    colors = [get_color(avg) for avg in averages]        


    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color=colors)

    # Add value labels on top of each bar
    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CAS Categorized Average Ratings', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with image and table data
    return JsonResponse({
        'image': image_base64,

    })
    
def plot_average_ratings_CCIS(request):
    if request.user.user_type != 'CCIS':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CCIS user
    
    # Query the database for ATYCB department data
    data = UploadCSV.objects.filter(dept_name='CCIS').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Define the categories and the corresponding question columns
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
    for category, questions in CATEGORIES.items():
        category_averages[category] = df[questions].mean().mean()

    # Prepare data for graph plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

     # Define color based on the average rating
    def get_color(value):
        if value < 1:  # Below threshold for red
            return 'red'
        elif value < 2: 
            return 'orange'
        elif value < 3:  
            return 'yellow'
        elif value < 4: 
            return '#9ACD32'                
        else:  # Above threshold for green
            return 'green'

    # Generate colors for each bar based on average ratings
    colors = [get_color(avg) for avg in averages]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color=colors)

    # Add value labels on top of each bar
    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CCIS Categorized Average Ratings', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Create a data structure for the hoverable table information
    tables = {
        "Presence/Guidance": [
            {"Indicator": "The instructor has set clear standards regarding their timeliness in responding to messages"},
            {"Indicator": "The instructor has provided the appropriate information and contact details for technical concern"},
            {"Indicator": "The instructor showed interest in student progress"}
        ],
        "Collaborative Learning": [
            {"Indicator": "The instructor encourages learners to participate"},
            {"Indicator": "The instructor implements Small Group Discussions (Breakout Rooms)"},
            {"Indicator": "The instructor provides equal opportunities for students to share ideas and viewpoints"}
        ],
        # Add other categories similarly...
    }

    # Return JSON response with image and table data
    return JsonResponse({
        'image': image_base64,
        'tables': tables
    })

def plot_average_ratings_CEA(request):
    if request.user.user_type != 'CEA':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CEA user    
    
    # Query the database for ATYCB department data
    data = UploadCSV.objects.filter(dept_name='CEA').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Define the categories and the corresponding question columns
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
    for category, questions in CATEGORIES.items():
        category_averages[category] = df[questions].mean().mean()

    # Prepare data for graph plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

    # Define color based on the average rating
    def get_color(value):
        if value < 1:  # Below threshold for red
            return 'red'
        elif value < 2: 
            return 'orange'
        elif value < 3:  
            return 'yellow'
        elif value < 4: 
            return '#9ACD32'                
        else:  # Above threshold for green
            return 'green'

    # Generate colors for each bar based on average ratings
    colors = [get_color(avg) for avg in averages]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color=colors)

    # Add value labels on top of each bar
    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CEA Categorized Average Ratings', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Create a data structure for the hoverable table information
    tables = {
        "Presence/Guidance": [
            {"Indicator": "The instructor has set clear standards regarding their timeliness in responding to messages"},
            {"Indicator": "The instructor has provided the appropriate information and contact details for technical concern"},
            {"Indicator": "The instructor showed interest in student progress"}
        ],
        "Collaborative Learning": [
            {"Indicator": "The instructor encourages learners to participate"},
            {"Indicator": "The instructor implements Small Group Discussions (Breakout Rooms)"},
            {"Indicator": "The instructor provides equal opportunities for students to share ideas and viewpoints"}
        ],
        # Add other categories similarly...
    }

    # Return JSON response with image and table data
    return JsonResponse({
        'image': image_base64,
        'tables': tables
    })
    
def plot_average_ratings_CHS(request):
    if request.user.user_type != 'CHS':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CHS user    
    
    # Query the database for ATYCB department data
    data = UploadCSV.objects.filter(dept_name='CHS').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Define the categories and the corresponding question columns
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
    for category, questions in CATEGORIES.items():
        category_averages[category] = df[questions].mean().mean()

    # Prepare data for graph plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

    # Define color based on the average rating
    def get_color(value):
        if value < 1:  # Below threshold for red
            return 'red'
        elif value < 2: 
            return 'orange'
        elif value < 3:  
            return 'yellow'
        elif value < 4: 
            return '#9ACD32'                
        else:  # Above threshold for green
            return 'green'

    # Generate colors for each bar based on average ratings
    colors = [get_color(avg) for avg in averages]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color=colors)

    # Add value labels on top of each bar
    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CHS Categorized Average Ratings', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Create a data structure for the hoverable table information
    tables = {
        "Presence/Guidance": [
            {"Indicator": "The instructor has set clear standards regarding their timeliness in responding to messages"},
            {"Indicator": "The instructor has provided the appropriate information and contact details for technical concern"},
            {"Indicator": "The instructor showed interest in student progress"}
        ],
        "Collaborative Learning": [
            {"Indicator": "The instructor encourages learners to participate"},
            {"Indicator": "The instructor implements Small Group Discussions (Breakout Rooms)"},
            {"Indicator": "The instructor provides equal opportunities for students to share ideas and viewpoints"}
        ],
        # Add other categories similarly...
    }

    # Return JSON response with image and table data
    return JsonResponse({
        'image': image_base64,
        'tables': tables
    })

def plot_average_ratings_NSTP(request):
    if request.user.user_type != 'NSTP':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not NSTP user
        
    # Query the database for ATYCB department data
    data = UploadCSV.objects.filter(dept_name='NSTP').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Define the categories and the corresponding question columns
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
    for category, questions in CATEGORIES.items():
        category_averages[category] = df[questions].mean().mean()

    # Prepare data for graph plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

    # Define color based on the average rating
    def get_color(value):
        if value < 1:  # Below threshold for red
            return 'red'
        elif value < 2: 
            return 'orange'
        elif value < 3:  
            return 'yellow'
        elif value < 4: 
            return '#9ACD32'                
        else:  # Above threshold for green
            return 'green'

    # Generate colors for each bar based on average ratings
    colors = [get_color(avg) for avg in averages]

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color=colors)

    # Add value labels on top of each bar
    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('NSTP Categorized Average Ratings', fontsize=20)
    plt.xticks(rotation=25, ha='right', fontsize=12)

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with image and table data
    return JsonResponse({
        'image': image_base64,
        'tables': tables
    })
    
# DEPARTMENT COMPARISON GRAPHS

# ATYCB & CAS GRAPH

# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def atycb_cas_comparison_view(request):
    if request.user.user_type != 'ATYCB, CAS':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not ATYCB & CAS user
        
    departments = ['ATYCB', 'CAS']  # Both ATYCB and CAS
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both ATYCB and CAS
    categories = list(department_averages['ATYCB'].keys())  # Using the categories from one department since both are the same
    averages_ATYCB = list(department_averages['ATYCB'].values())
    averages_CAS = list(department_averages['CAS'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for ATYCB and CAS
    colors_ATYCB = [get_color(value) for value in averages_ATYCB]
    colors_CAS = [get_color(value) for value in averages_CAS]

    # Plot for ATYCB with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_ATYCB, width=0.4, label='ATYCB', color=colors_ATYCB)

    # Plot for CAS with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_CAS, width=0.4, label='CAS', color=colors_CAS)

    # Add value labels on top of each bar for ATYCB
    for bar, value in zip(bars1, averages_ATYCB):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for CAS
    for bar, value in zip(bars2, averages_CAS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('ATYCB(left) vs CAS(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })

# ATYCB & CCIS GRAPH
# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def atycb_ccis_comparison_view(request):
    if request.user.user_type != 'ATYCB, CCIS':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not ATYCB & CCIS user    
    
    departments = ['ATYCB', 'CCIS']  # Both ATYCB and CCIS
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both ATYCB and CCIS
    categories = list(department_averages['ATYCB'].keys())  # Using the categories from one department since both are the same
    averages_ATYCB = list(department_averages['ATYCB'].values())
    averages_CCIS = list(department_averages['CCIS'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for ATYCB and CCIS
    colors_ATYCB = [get_color(value) for value in averages_ATYCB]
    colors_CCIS = [get_color(value) for value in averages_CCIS]

    # Plot for ATYCB with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_ATYCB, width=0.4, label='ATYCB', color=colors_ATYCB)

    # Plot for CCIS with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_CCIS, width=0.4, label='CCIS', color=colors_CCIS)

    # Add value labels on top of each bar for ATYCB
    for bar, value in zip(bars1, averages_ATYCB):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for CCIS
    for bar, value in zip(bars2, averages_CCIS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('ATYCB(left) vs CCIS(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })


# ATYCB & CEA GRAPH

# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def atycb_cea_comparison_view(request):
    if request.user.user_type != 'ATYCB, CEA':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not ATYCB & CEA user    
    
    departments = ['ATYCB', 'CEA']  # Both ATYCB and CEA
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both ATYCB and CEA
    categories = list(department_averages['ATYCB'].keys())  # Using the categories from one department since both are the same
    averages_ATYCB = list(department_averages['ATYCB'].values())
    averages_CEA = list(department_averages['CEA'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for ATYCB and CEA
    colors_ATYCB = [get_color(value) for value in averages_ATYCB]
    colors_CEA = [get_color(value) for value in averages_CEA]

    # Plot for ATYCB with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_ATYCB, width=0.4, label='ATYCB', color=colors_ATYCB)

    # Plot for CEA with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_CEA, width=0.4, label='CEA', color=colors_CEA)

    # Add value labels on top of each bar for ATYCB
    for bar, value in zip(bars1, averages_ATYCB):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for CEA
    for bar, value in zip(bars2, averages_CEA):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('ATYCB(left) vs CEA(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })


# ATYCB & CHS GRAPH

# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def atycb_chs_comparison_view(request):
    if request.user.user_type != 'ATYCB, CHS':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not ATYCB & CHS user
        
    departments = ['ATYCB', 'CHS']  # Both ATYCB and CHS
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both ATYCB and CHS
    categories = list(department_averages['ATYCB'].keys())  # Using the categories from one department since both are the same
    averages_ATYCB = list(department_averages['ATYCB'].values())
    averages_CHS = list(department_averages['CHS'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for ATYCB and CHS
    colors_ATYCB = [get_color(value) for value in averages_ATYCB]
    colors_CHS = [get_color(value) for value in averages_CHS]

    # Plot for ATYCB with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_ATYCB, width=0.4, label='ATYCB', color=colors_ATYCB)

    # Plot for CHS with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_CHS, width=0.4, label='CHS', color=colors_CHS)

    # Add value labels on top of each bar for ATYCB
    for bar, value in zip(bars1, averages_ATYCB):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for CHS
    for bar, value in zip(bars2, averages_CHS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('ATYCB(left) vs CHS(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals


    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })

# ATYCB & NSTP GRAPH

# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def atycb_nstp_comparison_view(request):
    if request.user.user_type not in ['ATYCB', 'NSTP']:
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not ATYCB or NSTP user
    
    departments = ['ATYCB', 'NSTP']  # Both ATYCB and NSTP
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both ATYCB and NSTP
    categories = list(department_averages['ATYCB'].keys())  # Using the categories from one department since both are the same
    averages_ATYCB = list(department_averages['ATYCB'].values())
    averages_NSTP = list(department_averages['NSTP'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for ATYCB and NSTP
    colors_ATYCB = [get_color(value) for value in averages_ATYCB]
    colors_NSTP = [get_color(value) for value in averages_NSTP]

    # Plot for ATYCB with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_ATYCB, width=0.4, label='ATYCB', color=colors_ATYCB)

    # Plot for NSTP with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_NSTP, width=0.4, label='NSTP', color=colors_NSTP)

    # Add value labels on top of each bar for ATYCB
    for bar, value in zip(bars1, averages_ATYCB):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for NSTP
    for bar, value in zip(bars2, averages_NSTP):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('ATYCB(left) vs NSTP(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })


# CAS & CCIS GRAPH

# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def cas_ccis_comparison_view(request):
    if request.user.user_type != 'CAS, CCIS':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CAS & CCIS user
        
    departments = ['CAS', 'CCIS']  # Both CAS and CCIS
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both CAS and CCIS
    categories = list(department_averages['CAS'].keys())  # Using the categories from one department since both are the same
    averages_CAS = list(department_averages['CAS'].values())
    averages_CCIS = list(department_averages['CCIS'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for CAS and CCIS
    colors_CAS = [get_color(value) for value in averages_CAS]
    colors_CCIS = [get_color(value) for value in averages_CCIS]

    # Plot for CAS with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_CAS, width=0.4, label='CAS', color=colors_CAS)

    # Plot for CCIS with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_CCIS, width=0.4, label='CCIS', color=colors_CCIS)

    # Add value labels on top of each bar for CAS
    for bar, value in zip(bars1, averages_CAS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for CCIS
    for bar, value in zip(bars2, averages_CCIS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CAS(left) vs CCIS(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })


# CAS & CEA GRAPH

# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def cas_cea_comparison_view(request):
    if request.user.user_type != 'CAS, CEA':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CAS & CEA user
        
    departments = ['CAS', 'CEA']  # Both CAS and CEA
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both CAS and CEA
    categories = list(department_averages['CAS'].keys())  # Using the categories from one department since both are the same
    averages_CAS = list(department_averages['CAS'].values())
    averages_CEA = list(department_averages['CEA'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for CAS and CEA
    colors_CAS = [get_color(value) for value in averages_CAS]
    colors_CEA = [get_color(value) for value in averages_CEA]

    # Plot for CAS with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_CAS, width=0.4, label='CAS', color=colors_CAS)

    # Plot for CEA with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_CEA, width=0.4, label='CEA', color=colors_CEA)

    # Add value labels on top of each bar for CAS
    for bar, value in zip(bars1, averages_CAS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for CEA
    for bar, value in zip(bars2, averages_CEA):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CAS(left) vs CEA(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })


# CAS & CHS GRAPH

# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def cas_chs_comparison_view(request):
    if request.user.user_type != 'CAS, CHS':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CAS & CHS user
        
    departments = ['CAS', 'CHS']  # Both CAS and CHS
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both CAS and CHS
    categories = list(department_averages['CAS'].keys())  # Using the categories from one department since both are the same
    averages_CAS = list(department_averages['CAS'].values())
    averages_CHS = list(department_averages['CHS'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for CAS and CHS
    colors_CAS = [get_color(value) for value in averages_CAS]
    colors_CHS = [get_color(value) for value in averages_CHS]

    # Plot for CAS with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_CAS, width=0.4, label='CAS', color=colors_CAS)

    # Plot for CHS with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_CHS, width=0.4, label='CHS', color=colors_CHS)

    # Add value labels on top of each bar for CAS
    for bar, value in zip(bars1, averages_CAS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for CHS
    for bar, value in zip(bars2, averages_CHS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CAS(left) vs CHS(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })



# CAS & NSTP GRAPH

# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def cas_nstp_comparison_view(request):
    if request.user.user_type != 'CAS, NSTP':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CAS & NSTP user
        
    departments = ['CAS', 'NSTP']  # Both CAS and NSTP
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both CAS and NSTP
    categories = list(department_averages['CAS'].keys())  # Using the categories from one department since both are the same
    averages_CAS = list(department_averages['CAS'].values())
    averages_NSTP = list(department_averages['NSTP'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for CAS and NSTP
    colors_CAS = [get_color(value) for value in averages_CAS]
    colors_NSTP = [get_color(value) for value in averages_NSTP]

    # Plot for CAS with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_CAS, width=0.4, label='CAS', color=colors_CAS)

    # Plot for NSTP with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_NSTP, width=0.4, label='NSTP', color=colors_NSTP)

    # Add value labels on top of each bar for CAS
    for bar, value in zip(bars1, averages_CAS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for NSTP
    for bar, value in zip(bars2, averages_NSTP):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CAS(left) vs NSTP(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })


# CCIS & CEA
# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def ccis_cea_comparison_view(request):
    if request.user.user_type != 'CCIS, CEA':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CCIS & CEA user    
    
    departments = ['CCIS', 'CEA']  # Both CCIS and CEA
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both CCIS and CEA
    categories = list(department_averages['CCIS'].keys())  # Using the categories from one department since both are the same
    averages_CCIS = list(department_averages['CCIS'].values())
    averages_CEA = list(department_averages['CEA'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for CCIS and CEA
    colors_CCIS = [get_color(value) for value in averages_CCIS]
    colors_CEA = [get_color(value) for value in averages_CEA]

    # Plot for CCIS with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_CCIS, width=0.4, label='CCIS', color=colors_CCIS)

    # Plot for CEA with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_CEA, width=0.4, label='CEA', color=colors_CEA)

    # Add value labels on top of each bar for CCIS
    for bar, value in zip(bars1, averages_CCIS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for CEA
    for bar, value in zip(bars2, averages_CEA):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CCIS(left) vs CEA(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })



# CCIS & CHS
# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def ccis_chs_comparison_view(request):
    if request.user.user_type != 'CCIS, CHS':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CCIS & CHS user    
    
    departments = ['CCIS', 'CHS']  # Both CCIS and CHS
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both CCIS and CHS
    categories = list(department_averages['CCIS'].keys())  # Using the categories from one department since both are the same
    averages_CCIS = list(department_averages['CCIS'].values())
    averages_CHS = list(department_averages['CHS'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for CCIS and CHS
    colors_CCIS = [get_color(value) for value in averages_CCIS]
    colors_CHS = [get_color(value) for value in averages_CHS]

    # Plot for CCIS with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_CCIS, width=0.4, label='CCIS', color=colors_CCIS)

    # Plot for CHS with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_CHS, width=0.4, label='CHS', color=colors_CHS)

    # Add value labels on top of each bar for CCIS
    for bar, value in zip(bars1, averages_CCIS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for CHS
    for bar, value in zip(bars2, averages_CHS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CCIS(left) vs CHS(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })


# CCIS & NSTP
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def ccis_nstp_comparison_view(request):
    if request.user.user_type != 'CCIS, NSTP':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CCIS & NSTP user
       
    departments = ['CCIS', 'NSTP']  # Both CCIS and NSTP
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both CCIS and NSTP
    categories = list(department_averages['CCIS'].keys())  # Using the categories from one department since both are the same
    averages_CCIS = list(department_averages['CCIS'].values())
    averages_NSTP = list(department_averages['NSTP'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for CCIS and NSTP
    colors_CCIS = [get_color(value) for value in averages_CCIS]
    colors_NSTP = [get_color(value) for value in averages_NSTP]

    # Plot for CCIS with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_CCIS, width=0.4, label='CCIS', color=colors_CCIS)

    # Plot for NSTP with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_NSTP, width=0.4, label='NSTP', color=colors_NSTP)

    # Add value labels on top of each bar for CCIS
    for bar, value in zip(bars1, averages_CCIS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for NSTP
    for bar, value in zip(bars2, averages_NSTP):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CCIS(left) vs NSTP(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })


# CEA & CHS
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def cea_chs_comparison_view(request):
    if request.user.user_type != 'CEA, CHS':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CEA & CHS user
        
    departments = ['CEA', 'CHS']  # Both CEA and CHS
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both CEA and CHS
    categories = list(department_averages['CEA'].keys())  # Using the categories from one department since both are the same
    averages_CEA = list(department_averages['CEA'].values())
    averages_CHS = list(department_averages['CHS'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for CEA and CHS
    colors_CEA = [get_color(value) for value in averages_CEA]
    colors_CHS = [get_color(value) for value in averages_CHS]

    # Plot for CEA with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_CEA, width=0.4, label='CEA', color=colors_CEA)

    # Plot for CHS with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_CHS, width=0.4, label='CHS', color=colors_CHS)

    # Add value labels on top of each bar for CEA
    for bar, value in zip(bars1, averages_CEA):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for CHS
    for bar, value in zip(bars2, averages_CHS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CEA(left) vs CHS(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })


# CEA & NSTP

# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def cea_nstp_comparison_view(request):
    if request.user.user_type != 'CEA, NSTP':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CEA & NSTP user
        
    departments = ['CEA', 'NSTP']  # Both CEA and NSTP
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both CEA and NSTP
    categories = list(department_averages['CEA'].keys())  # Using the categories from one department since both are the same
    averages_CEA = list(department_averages['CEA'].values())
    averages_NSTP = list(department_averages['NSTP'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for CEA and NSTP
    colors_CEA = [get_color(value) for value in averages_CEA]
    colors_NSTP = [get_color(value) for value in averages_NSTP]

    # Plot for CEA with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_CEA, width=0.4, label='CEA', color=colors_CEA)

    # Plot for NSTP with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_NSTP, width=0.4, label='NSTP', color=colors_NSTP)

    # Add value labels on top of each bar for CEA
    for bar, value in zip(bars1, averages_CEA):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for NSTP
    for bar, value in zip(bars2, averages_NSTP):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CEA(left) vs NSTP(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })


# CHS & NSTP
# Define your get_color function above the view
def get_color(value):
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2:  # Between thresholds for orange
        return 'orange'
    elif value < 3:  # Between thresholds for yellow
        return 'yellow'
    elif value < 4:  # Between thresholds for yellowgreen
        return '#9ACD32'                
    else:  # Above threshold for green
        return 'green'

# Your existing view function
def chs_nstp_comparison_view(request):
    if request.user.user_type != 'CHS, NSTP':
        return HttpResponseForbidden("You do not have permission to access this data.")  # Deny access if not CHS & NSTP user    
    
    departments = ['CHS', 'NSTP']  # Both CHS and NSTP
    department_averages = {}

    for dept in departments:
        # Query the database for department data
        data = UploadCSV.objects.filter(dept_name=dept).values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

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
        category_averages = {category: df[questions].mean().mean() for category, questions in CATEGORIES.items()}
        department_averages[dept] = category_averages

    # Plot the bar graph for both CHS and NSTP
    categories = list(department_averages['CHS'].keys())  # Using the categories from one department since both are the same
    averages_CHS = list(department_averages['CHS'].values())
    averages_NSTP = list(department_averages['NSTP'].values())

    plt.figure(figsize=(15, 12))

    # Set colors based on the average ratings for CHS and NSTP
    colors_CHS = [get_color(value) for value in averages_CHS]
    colors_NSTP = [get_color(value) for value in averages_NSTP]

    # Plot for CHS with respective colors
    bars1 = plt.bar([i - 0.2 for i in range(len(categories))], averages_CHS, width=0.4, label='CHS', color=colors_CHS)

    # Plot for NSTP with respective colors
    bars2 = plt.bar([i + 0.2 for i in range(len(categories))], averages_NSTP, width=0.4, label='NSTP', color=colors_NSTP)

    # Add value labels on top of each bar for CHS
    for bar, value in zip(bars1, averages_CHS):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add value labels on top of each bar for NSTP
    for bar, value in zip(bars2, averages_NSTP):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=14)

    # Add grid lines and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.title('CHS(left) vs NSTP(right) Average Ratings', fontsize=20)
    plt.xticks(range(len(categories)), categories, rotation=25, ha='right', fontsize=12)

    # Set y-axis ticks with 0.5 intervals
    plt.yticks([i * 0.5 for i in range(9)])  # Y-axis from 0 to 4.5 with 0.5 intervals

    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image as base64
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response with the image
    return JsonResponse({
        'image': image_base64
    })


# INSTRUCTOR AVERAGE RATINGS

# Define categories and their associated questions
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
    # Query the database for the selected instructor
    data = UploadCSV.objects.filter(resp_fac=instructor_name).values(
        *[f"question_{i}" for i in range(1, 31)]
    )
    df = pd.DataFrame(data)

    # Convert question columns to numeric and calculate category averages
    df = df.apply(pd.to_numeric, errors="coerce")

    category_averages = {
        category: df[questions].mean().mean() for category, questions in CATEGORIES.items()
    }

    # Prepare data for plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color="#2196f3", label="Average Rating")

    # Add value labels on top of each bar
    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                 ha="center", va="bottom", fontsize=14)

    # Add labels and gridlines
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlabel("Categories", fontsize=14)
    plt.ylabel("Average Rating", fontsize=14)
    plt.title(f"Categorized Ratings for {instructor_name}", fontsize=20)
    plt.xticks(rotation=25, ha="right", fontsize=12)
    plt.legend()

    # Save plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()

    image_base64 = base64.b64encode(img.getvalue()).decode("utf-8")

    return JsonResponse({"image": image_base64})


# TERMS TREND GRAPH

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

def plot_term_trend(term_filter):
    df = get_terms_data()

    if df is None:
        return JsonResponse({'error': 'No data available'}, status=400)

    # Filter the DataFrame based on the selected term
    filtered_df = df[df['term'] == term_filter]

    if filtered_df.empty:
        return JsonResponse({'error': f'Term "{term_filter}" not available'}, status=400)

    # Calculate the overall score as the average of all question_* fields
    filtered_df['overall_score'] = filtered_df.iloc[:, 1:].mean(axis=1)

    # Calculate average overall scores per term
    term_average = filtered_df['overall_score'].mean()

    # Prepare data for graph plotting
    terms = [term_filter]  # List of terms for the graph
    averages = [term_average]  # Average for the selected term

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(terms, averages, color='blue', label='Average Rating')

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

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(terms, averages, color='blue', label='Average Rating')

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
# Define your keywords for sentiment analysis
POSITIVE_KEYWORDS = ['good', 'happy', 'great', 'excellent', 'fantastic', 'love', 'amazing', 'approachable', 'learned', 'excellent', 'wonderful', 'Necessary learnings', 'Wonderful', 'Great teaching', 'Helps me a lot', 'Excellent at teaching', 'Easy to digest', 'Important skills', 'Fun and interactive', 'Socially engaging',
                     'Motivates', 'Encourages', 'Effective', 'Clear examples', 'Competent', 'Fun activities', 'Engaging', 'Interesting', 'Patient and kind', 'Welcoming atmosphere', 'Encouraging participation', 'Makes learning interesting', 'Approachable', 'Calm', 'Good at teaching', 'Entertaining', 'Accommodating', 
                     'Supports inclusivity', 'Responsive', 'Exceptional instructor', 'Interesting discussions', 'Holistic development', 'Valuable skills', 'Smooth classes', 'Well-structured', 'Positive learning environment', 'Dedicated', 'Warm and inclusive', 'Comfortable asking questions', 'Fascinating', 'Efficient and relevant', 
                     'Fun yet professional', 'Challenging yet informative', 'Interesting topics', 'commendable', 'helpful', 'fun', 'skilled', 'commendable', 'interactive', 'relatable', 'effective', 'enjoyable', 'nice', 'passionate', 'understandable', 'challenging', 'cheerful', 'interesting', 'helpful', 'professional', 
                     'engaging', 'splendid', 'excellent', 'fascinating', 'informative', 'beneficial', 'finest', 'eloquence', 'humility', 'engagement', 'participate', 'innovative', 'collaborate', 'active', 'professional', 'transformative', 'enriching', 'critical thinking', 'prepared', 'understanding', 'approachable', 
                     'thoughtful', 'master at teaching', 'sweet', 'considerate', 'uplifting', 'incredible', 'support', 'inspiration', 'kind', 'accommodating', 'patient', 'supportive', 'enjoyable', 'exceptional', 'practical', 'friendly', 'attentive', 'easy-going', 'consistent', 'ideal professor', 'very considerate', 
                     'making the lesson easier', 'very well', 'teaches very well', 'Thank you', 'good teacher', 'kind', 'knowledgeable', 'teaches well', 'makes it easier', 'bearable', 'Approachable', 'expert', 'not so difficult', 'good instructor', 'Kind', 'effectiveness', 'clarity', 'hard course']
NEGATIVE_KEYWORDS = ['bad', 'sad', 'terrible', 'hate', 'awful', 'poor', 'dissatisfied', 'Social anxiety', 'Nervous', 'Anxious', 'Tiring', 'Challenging', 'Redundant', 'Confusing', 'overwhelming', 'too much', 'confusing', 'challenging', 'draining', 'difficult', 'unsure', 'excessive', 'frustrating', 'hard to stay on track', 'struggling']

# Exclusion pattern
exclusion_pattern = r'^(None|none|NONE\.|N/A|(N/A)|(n/a)|n\.a\.|NONE|NADA|#NAME\?|\.{1,2}|[:;][-]?[)(]+|[-.]*|Meh\.|Ok|Okay|Okay\.|okay\.|okay|Ambot|ambot|nope|Klw|none|nome|N/A\.|None\.|None!|none\.|Nome|Wala|wala|hm|ok|N/A|nonw| ,)$'

def comments_table_view(request):
    # Get the filter parameters from the request
    term_filter = request.GET.get('term', 'all')  # Default to 'all' if no term is selected
    comment_type_filter = request.GET.get('comment_type', 'both')  # Default to 'both'
    sentiment_filter = request.GET.get('sentiment', 'all')  # Default to 'all'

    # Base query: filter by term if selected
    if term_filter == 'term1':
        comments = UploadCSV.objects.filter(term='Term 1').values('question_31', 'question_32', 'term')
    elif term_filter == 'term2':
        comments = UploadCSV.objects.filter(term='Term 2').values('question_31', 'question_32', 'term')
    else:
        comments = UploadCSV.objects.values('question_31', 'question_32', 'term')

    # Filter comments by type and length
    if comment_type_filter == 'instructor':
        comments = comments.exclude(
            Q(question_31__regex=exclusion_pattern) | 
            Q(question_31__isnull=True) | 
            Q(question_31__exact='')
        ).annotate(comment_length=Length('question_31')).filter(comment_length__gte=4)
    elif comment_type_filter == 'course':
        comments = comments.exclude(
            Q(question_32__regex=exclusion_pattern) | 
            Q(question_32__isnull=True) | 
            Q(question_32__exact='')
        ).annotate(comment_length=Length('question_32')).filter(comment_length__gte=4)
    else:  # for 'both'
        comments = comments.annotate(
            instructor_length=Length('question_31'),
            course_length=Length('question_32')
        ).exclude(
            Q(question_31__regex=exclusion_pattern) | 
            Q(question_31__isnull=True) | 
            Q(question_31__exact='') |
            Q(question_32__regex=exclusion_pattern) | 
            Q(question_32__isnull=True) | 
            Q(question_32__exact='')
        ).filter(instructor_length__gte=4, course_length__gte=4)

    # Classify comments based on keywords
    for comment in comments:
        comment['sentiment'] = 'neutral'  # Default sentiment
        if comment_type_filter in ['instructor', 'both']:
            instructor_comment = comment['question_31'].lower()
            if any(keyword in instructor_comment for keyword in POSITIVE_KEYWORDS):
                comment['sentiment'] = 'good'
            elif any(keyword in instructor_comment for keyword in NEGATIVE_KEYWORDS):
                comment['sentiment'] = 'bad'

        if comment_type_filter in ['course', 'both']:
            course_comment = comment['question_32'].lower()
            if any(keyword in course_comment for keyword in POSITIVE_KEYWORDS):
                comment['sentiment'] = 'good'
            elif any(keyword in course_comment for keyword in NEGATIVE_KEYWORDS):
                comment['sentiment'] = 'bad'

    # Filter comments based on sentiment
    if sentiment_filter == 'good':
        comments = [comment for comment in comments if comment['sentiment'] == 'good']
    elif sentiment_filter == 'bad':
        comments = [comment for comment in comments if comment['sentiment'] == 'bad']

    # Print the comments for debugging
    print(list(comments))
    
    # Return the filtered comments as a JSON response
    return JsonResponse({
        'comments': list(comments)
    })


# INSTRUCTOR RANKING
    
def get_color(value):
    """Returns a color based on the score value."""
    if value < 1:  # Below threshold for red
        return 'red'
    elif value < 2: 
        return 'orange'
    elif value < 3:  
        return 'yellow'
    elif value < 4: 
        return '#9ACD32'  # Light green                
    else:  # Above threshold for green
        return 'green'

def generate_instructor_graph(department_selected=None, term_selected=None, survey_name=None):
    instructors = UploadCSV.objects.all()

    print(f"Initial number of instructors: {instructors.count()}")

    if department_selected and department_selected != 'all':
        instructors = instructors.filter(dept_name=department_selected)
        print(f"Filtered by department: {department_selected} - Number of instructors: {instructors.count()}")

    # Filter by survey_name if provided
    if survey_name and survey_name != 'all':
        instructors = instructors.filter(survey_name=survey_name)  # Filter by survey_name
        print(f"Filtered by survey name: {survey_name} - Number of instructors: {instructors.count()}")

    if not instructors.exists():
        print("No instructors found for the selected filters.")
        return None

    # Initialize dictionary to hold scores
    average_scores = {}
    
    for instructor in instructors:
        instructor_name = instructor.resp_fac
        total_score = 0
        question_count = 30

        # Calculate total score from question_1 to question_30
        for i in range(1, question_count + 1):
            score = getattr(instructor, f'question_{i}', None)
            if score is None:
                print(f"Missing score for instructor {instructor_name}, question {i}")  # Debugging
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
    leaderboard.sort(key=lambda x: x[1], reverse=True)  # Sort by average score in descending order

    # Limit to top 10 instructors
    top_leaderboard = leaderboard[:10]
    top_names = [name for name, _ in top_leaderboard]
    top_averages = [average for _, average in top_leaderboard]

    # Reverse the order of names and averages for the highest to be at the top
    top_names.reverse()
    top_averages.reverse()

    # Determine colors for the bars
    bar_colors = [get_color(avg) for avg in top_averages]

    # Plot the leaderboard with increased width
    plt.figure(figsize=(15, 8))  # Adjusted width and height
    bars = plt.barh(top_names, top_averages, color=bar_colors)

    # Set titles and labels with increased font size
    plt.title('Top 10 Instructor Leaderboard', fontsize=20)  # Increased title font size
    plt.xlabel('Average Rating', fontsize=16)  # Increased x-axis label font size
    plt.ylabel('Instructors', fontsize=16)  # Increased y-axis label font size

    # Increase tick label font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Annotate the bars with their corresponding values
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.2f}',
                 va='center', ha='left', fontsize=12)

    plt.tight_layout()

    # Save it to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Encode it to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@require_GET
def instructor_ranking_graph(request):
    department_selected = request.GET.get('dept_name', None)
    survey_name = request.GET.get('survey_name', None)  # Get survey name from request

    # Add logging for debugging purposes
    print(f'Request received. Department Selected: {department_selected}, Survey Name Selected: {survey_name}')

    # Generate the graph
    graph_image = generate_instructor_graph(department_selected, survey_name)

    # Handle case where no graph is generated
    if graph_image is None:
        print("Graph could not be generated, no data available.")
        return JsonResponse({'error': 'No data found for the selected filters'}, status=400)

    print("Graph generated successfully.")
    return JsonResponse({'graph': graph_image})