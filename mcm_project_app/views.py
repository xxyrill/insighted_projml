import matplotlib
import plotly.express as px
import plotly.graph_objects as go
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64
import re
import numpy as np
from io import BytesIO

matplotlib.use('Agg')  # Set the backend to Agg for non-interactive use
from django.http import HttpResponse
from django.conf import settings
from .utilities import csvPathFileName
from django.shortcuts import render
from .utils import upload_csv_to_db
from django.contrib import messages
from io import BytesIO
from nltk.sentiment import SentimentIntensityAnalyzer
from django.db import IntegrityError
from nltk.corpus import stopwords
from django.http import JsonResponse
from accounts.models import UploadCSV
from django.core.cache import cache
from django.shortcuts import render

def login_page(request):
    return render(request, 'authentications/login.html')

# Delete database
def calculate_sentiment_scores():
    # Initialize the Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    # Step 1: Query the UploadCSV table to get comments data
    data = UploadCSV.objects.all().values('question_31', 'question_32')  # Adjust these fields as needed

    # Step 2: Convert the QuerySet to a DataFrame for easier manipulation
    df = pd.DataFrame(list(data))

    # Step 3: Calculate sentiment scores for comments
    df['sentiment_score_question_31'] = df['question_31'].apply(lambda x: sia.polarity_scores(str(x))['compound'] if pd.notnull(x) else 0)
    df['sentiment_score_question_32'] = df['question_32'].apply(lambda x: sia.polarity_scores(str(x))['compound'] if pd.notnull(x) else 0)

    # Step 4: Return DataFrame with sentiment scores
    return df[['sentiment_score_question_31', 'sentiment_score_question_32']]

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
                crs_name=row['crs_name']
            )
        except IntegrityError as e:
            print(f"Error inserting row: {e}")  # Log or handle the error if necessary

    # Step 4: Extract unique values for filters
    departments = df['deptname'].unique()
    instructors = df['resp_fac'].unique()
    courses = df['crs_name'].unique()

    print("Unique Departments:", departments)
    print("Unique Instructors:", instructors)
    print("Unique Courses:", courses)

    return departments, instructors, courses

def dashboard_view(request):
    departments = list(UploadCSV.objects.values_list('dept_name', flat=True).distinct())  # Convert QuerySet to list
    instructors = list(UploadCSV.objects.values_list('resp_fac', flat=True).distinct())  # Convert QuerySet to list
    courses = list(UploadCSV.objects.values_list('crs_name', flat=True).distinct())  # Convert QuerySet to list

    context = {
        'departments': departments,
        'instructors': instructors,
        'courses': courses,
        'user_type': request.user.user_type,
        'messages': messages.get_messages(request),  # If you have messages
    }

    return render(request, 'dashboard/dashboard.html', context)

def upload_csv(request):
    if request.method == 'POST':
        csv_file = request.FILES['csv_file']
        csv_file_path = f"{settings.MEDIA_ROOT}/{csv_file.name}"

        # Step 1: Save the uploaded file
        with open(csv_file_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)

        # Step 2: Call the function to upload CSV to database
        try:
            # Step 3: Get unique values after uploading CSV
            departments, instructors, courses = upload_csv_to_db(csv_file_path)

            # Set success message
            messages.success(request, "CSV uploaded successfully.")

            # Step 4: Render the dashboard with updated context
            context = {
                'departments': departments.tolist(),  # Convert to list for template rendering
                'instructors': instructors.tolist(),  # Convert to list for template rendering
                'courses': courses.tolist(),          # Convert to list for template rendering
                'user_type': request.user.user_type    # Pass user type if needed
            }
            return render(request, 'dashboard.html', context)

        except Exception as e:
            messages.error(request, f"Error uploading CSV: {e}")
            return render(request, 'upload_csv.html')  # Render your upload form

    return render(request, 'upload_csv.html')  # Render your upload form

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

    return departments, instructors, courses


def render_filter_page(request):
    """View to render the filter page with populated dropdowns for Department, Instructor, and Courses."""
    # Get the unique filter options
    departments, instructors, courses = get_unique_filter_options()

    # Pass the unique options to the template context
    context = {
        'departments': departments,
        'instructors': instructors,
        'courses': courses
    }
    return render(request, 'dashboard.html', context)


#4. Pie Chart Comments
nltk.download('stopwords')
# Function to get word frequencies

### DEPARTMENT AVERAGE RATINGS
def plot_average_ratings_ATYCB(request):
    # Check if data is cached
    data = cache.get('average_ratings_data')

    if not data:
        # Query the database for ATYCB department data
        data = UploadCSV.objects.filter(dept_name='ATYCB').values(
            'resp_fac', *['question_{}'.format(i) for i in range(1, 31)]
        )
        df = pd.DataFrame(list(data))

        # Convert question columns to numeric and handle NaN values
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Define categories and corresponding questions
        CATEGORIES = {
            "Presence/Guidance": ['question_1', 'question_2', 'question_3'],
            "Collaborative Learning": ['question_4', 'question_5', 'question_6',
                                       'question_7', 'question_8', 'question_9'],
            "Active Learning": ['question_10', 'question_11', 'question_12'],
            "Content Knowledge and Proficiency": ['question_13', 'question_14', 'question_15'],
            "Course Expectations": ['question_16', 'question_17', 'question_18'],
            "Clarity/Instructions": ['question_19', 'question_20', 'question_21'],
            "Feedback": ['question_22', 'question_23', 'question_24'],
            "Inclusivity": ['question_25', 'question_26', 'question_27'],
            "Outcome-Based Education": ['question_28', 'question_29', 'question_30']
        }

        # Calculate average ratings for each category
        category_averages = {
            category: df[questions].mean().mean()
            for category, questions in CATEGORIES.items()
            if not df[questions].isnull().all(axis=1).any()
        }

        # Store processed data in cache (15 minutes)
        cache.set('average_ratings_data', category_averages, timeout=60 * 15)
    else:
        category_averages = data

    # Hover text for categories
    hover_text = {
        "Presence/Guidance": """
        <b>I. Presence/Guidance</b><br>
        1. The instructor has set clear standards regarding their timeliness in responding to messages<br>
        2. The instructor has provided the appropriate information and contact details for technical concerns<br>
        3. The instructor showed interest in student progress
        """,
        "Collaborative Learning": """
        <b>II. Collaborative Learning</b><br>
        a. SYNCHRONOUS:<br>
        i. Encourages participation<br>
        ii. Implements small group discussions (breakout rooms)<br>
        iii. Provides equal opportunities for students to share ideas<br>
        b. ASYNCHRONOUS:<br>
        i. Requires participation<br>
        ii. Provides platforms for small group discussions<br>
        iii. Tasks require collaboration
        """,
        # Add similar hover text for other categories...
    }

    # Prepare data for plotting
    categories = list(category_averages.keys())
    averages = list(category_averages.values())

    # Create a Plotly bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=averages,
                text=[f'{value:.2f}' for value in averages],  # Display values
                hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}<br>%{customdata}<extra></extra>",
                customdata=[hover_text.get(cat, "No details available.") for cat in categories],
                marker=dict(color='blue')
            )
        ]
    )

    fig.update_layout(
        title='ATYCB Categorized Average Ratings',
        xaxis_title='Categories',
        yaxis_title='Average Rating',
        hovermode='x'
    )

    # Save plot as image in BytesIO object
    img = BytesIO()
    fig.write_image(img, format='png', engine='kaleido')
    img.seek(0)
    graph_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return the image through the API endpoint
    return JsonResponse({'image': graph_base64})

def plot_average_ratings_CAS(request):
    # Query the database for ATYCB department data
    data = UploadCSV.objects.filter(dept_name='CAS').values('resp_fac', *['question_{}'.format(i) for i in range(1, 31)])
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

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color='blue', label='Average Rating')

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
    plt.legend()

    # Save the plot to a BytesIO object and encode it as a base64 string
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Return image data as a base64 string
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response
    return JsonResponse({'image': image_base64})

def plot_average_ratings_CCIS(request):
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

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color='blue', label='Average Rating')

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
    plt.legend()

    # Save the plot to a BytesIO object and encode it as a base64 string
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Return image data as a base64 string
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response
    return JsonResponse({'image': image_base64})

def plot_average_ratings_CEA(request):
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

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color='blue', label='Average Rating')

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
    plt.legend()

    # Save the plot to a BytesIO object and encode it as a base64 string
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Return image data as a base64 string
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response
    return JsonResponse({'image': image_base64})

def plot_average_ratings_CHS(request):
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

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color='blue', label='Average Rating')

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
    plt.legend()

    # Save the plot to a BytesIO object and encode it as a base64 string
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Return image data as a base64 string
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response
    return JsonResponse({'image': image_base64})

def plot_average_ratings_NSTP(request):
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

    # Plot the bar graph
    plt.figure(figsize=(15, 12))
    bars = plt.bar(categories, averages, color='blue', label='Average Rating')

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
    plt.legend()

    # Save the plot to a BytesIO object and encode it as a base64 string
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Return image data as a base64 string
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return JSON response
    return JsonResponse({'image': image_base64})

### INSTRUCTOR AVERAGE RATINGS

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
# Function to plot a pie chart and convert it to base64
def plot_pie_chart(word_counts, title):
    labels, sizes = zip(*word_counts)
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    plt.tight_layout()

    # Save plot to BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Convert image to base64 string
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

# View to generate pie charts and return them as base64 images in JSON response
def plot_comments_pie_chart(request):
    comments_instructor = UploadCSV.objects.values_list('question_31', flat=True)
    comments_course = UploadCSV.objects.values_list('question_32', flat=True)

    # Get word frequencies
    instructor_word_counts = get_word_frequencies(comments_instructor)
    course_word_counts = get_word_frequencies(comments_course)

    # Generate pie charts and encode them to base64
    instructor_chart_base64 = plot_pie_chart(instructor_word_counts, 'Instructor Comments Word Frequencies')
    course_chart_base64 = plot_pie_chart(course_word_counts, 'Course Comments Word Frequencies')

    # Return the base64-encoded images in a JSON response
    return JsonResponse({
        'instructor_chart': instructor_chart_base64,
        'course_chart': course_chart_base64
    })
# 5. Length of Comments Analysis
def plot_length_of_comments_analysis(request):
    # Step 1: Query the UploadCSV table to get comments data
    data = UploadCSV.objects.all().values('question_31', 'question_32')  # Adjust these fields as needed

    # Step 2: Convert the QuerySet to a DataFrame for easier manipulation
    df = pd.DataFrame(list(data))

    # Step 3: Calculate lengths of comments
    df['instructor_comment_length'] = df['question_31'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
    df['course_comment_length'] = df['question_32'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

    # Step 4: Ensure the dataframe is not empty
    if df.empty:
        return HttpResponse("No data available to analyze comment lengths.", status=400)

    # Step 5: Generate the histogram plot
    plt.figure(figsize=(12, 10))

    # Increase the number of bins to reduce congestion
    ax = sns.histplot(df['instructor_comment_length'], bins=30, color='blue', kde=True,
                      label='Instructor Comments', alpha=0.5)
    sns.histplot(df['course_comment_length'], bins=30, color='orange', kde=True,
                  label='Course Comments', alpha=0.5)

    plt.title('Distribution of Comment Lengths')
    plt.xlabel('Comment Length (characters)')
    plt.ylabel('Frequency')
    plt.legend()

    # Step 6: Add gridlines and display counts on top of bars
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines

    # Add counts on top of bars
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:  # Check to avoid division by zero
            plt.text(patch.get_x() + patch.get_width() / 2, height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Convert image to base64 string
    histogram_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 7: Prepare the context with the image
    context = {
        'histogram': histogram_base64
    }

    # Step 8: Render the template
    return JsonResponse({'image': histogram_base64})


# 14. Instructor Leaderboard
def plot_instructor_leaderboard(request):
    # Query the UploadCSV table for necessary data
    data = UploadCSV.objects.all().values('resp_fac', *['question_{}'.format(i) for i in range(1, 30)])
    df = pd.DataFrame(list(data))

    # Convert question columns to numeric, coercing errors to NaN
    question_columns = ['question_{}'.format(i) for i in range(1, 30)]
    df[question_columns] = df[question_columns].apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values in question columns
    df.dropna(subset=question_columns, inplace=True)

    # Calculate average ratings for each instructor
    avg_ratings = df[question_columns].mean(axis=1)
    df['average_rating'] = avg_ratings

    # Sort instructors by average rating
    leaderboard = df.groupby('resp_fac')['average_rating'].mean().reset_index().sort_values(by='average_rating',
                                                                                            ascending=False)

    # Limit to top 10 instructors to reduce congestion (optional)
    leaderboard = leaderboard.head(10)

    # Plot the leaderboard with increased width
    plt.figure(figsize=(15, 8))  # Adjusted width and height
    bars = plt.barh(leaderboard['resp_fac'], leaderboard['average_rating'], color='skyblue')

    # Set titles and labels with increased font size
    plt.title('Instructor Leaderboard', fontsize=20)  # Increased title font size
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

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image to base64 to return it in HTML
    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return the image in JSON format
    return JsonResponse({'image': image_base64})


