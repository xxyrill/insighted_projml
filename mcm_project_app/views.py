import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for non-interactive use
import matplotlib.pyplot as plt
import pandas as pd
import io
import seaborn as sns
from django.http import HttpResponse
from django.conf import settings
import os
from .utilities import csvPathFileName
from wordcloud import WordCloud
from textblob import TextBlob
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
import base64
import chardet
import dask.dataframe as dd
from accounts.models import UploadCSV
from .utils import upload_csv_to_db
from django.db import connection
from django.contrib import messages


def login_page(request):
    return render(request, 'authentications/login.html')

def upload_csv_to_db(csv_file_path):
    df = pd.read_csv(csv_file_path)
    print("DataFrame Loaded:\n", df)

    # Clear existing data if necessary
    with connection.cursor() as cursor:
        cursor.execute("TRUNCATE TABLE UploadCSV")  # Use the actual table name

    # Insert data into your table
    for _, row in df.iterrows():
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO your_actual_table_name (dept_name, resp_fac, crs_name) VALUES (%s, %s, %s)",
                (row['deptname'], row['resp_fac'], row['crs_name'])
            )

    # Extract unique values for filters
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

        # Save the uploaded file
        with open(csv_file_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)

        # Call the function to upload CSV to database
        try:
            # Get unique values after uploading CSV
            departments, instructors, courses = upload_csv_to_db(csv_file_path)

            # Set success message
            messages.success(request, "CSV uploaded successfully.")

            # Render the dashboard with updated context
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
        img = plot_department_avg_ratings(request)
    # Add logic for other graphs as needed

    # Convert image to base64 for frontend display
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return JsonResponse({'image': img_base64}, safe=False)


def get_unique_filter_options():
    """Function to extract unique filter options for Department, Instructor, and Courses."""
    csv_file_path = csvPathFileName()
    data = pd.read_csv(csv_file_path)

    # Extract unique values for Department, Instructor, and Courses
    departments = data['deptname'].dropna().unique()  # Assuming 'deptname' is the column name for Department
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


#####


def plot_ratings_trend(request):
    # Step 1: Query the UploadCSV table to get the required data
    data = UploadCSV.objects.all().values('crs_year', 'question_1', 'question_2', 'question_3', 'question_4',
                                          'question_5', 'question_6', 'question_7', 'question_8', 'question_9',
                                          'question_10', 'question_11', 'question_12', 'question_13', 'question_14',
                                          'question_15', 'question_16', 'question_17', 'question_18', 'question_19',
                                          'question_20', 'question_21', 'question_22', 'question_23', 'question_24',
                                          'question_25', 'question_26', 'question_27', 'question_28', 'question_29',
                                          'question_30', 'question_31', 'question_32')

    # Step 2: Convert the QuerySet to a DataFrame for easier manipulation
    import pandas as pd
    df = pd.DataFrame(list(data))

    # Step 3: Ensure the dataframe is not empty
    if df.empty:
        return HttpResponse("No data available to generate the graph.", status=400)

    # Step 4: Convert all question columns to numeric, replacing errors with NaN
    question_columns = ['question_1', 'question_2', 'question_3', 'question_4', 'question_5',
                        'question_6', 'question_7', 'question_8', 'question_9', 'question_10',
                        'question_11', 'question_12', 'question_13', 'question_14', 'question_15',
                        'question_16', 'question_17', 'question_18', 'question_19', 'question_20',
                        'question_21', 'question_22', 'question_23', 'question_24', 'question_25',
                        'question_26', 'question_27', 'question_28', 'question_29', 'question_30',
                        'question_31', 'question_32',]

    df[question_columns] = df[question_columns].apply(pd.to_numeric, errors='coerce')
    df[question_columns] = df[question_columns].fillna(0)

    # Step 5: Calculate total ratings per row by summing the question columns
    df['total_rating'] = df[question_columns].sum(axis=1)

    # Step 6: Group by 'crs_year' and calculate the average total rating
    avg_ratings = df.groupby('crs_year')['total_rating'].mean().reset_index()

    # Step 7: Plot the average ratings trend over terms
    plt.figure(figsize=(10, 6))
    plt.plot(avg_ratings['crs_year'], avg_ratings['total_rating'], marker='o', linestyle='-', color='b')
    plt.title('Trend of Average Ratings Over Terms')
    plt.xlabel('Academic Term')
    plt.ylabel('Average Rating')
    plt.grid(True)

    # Step 8: Save the plot to a BytesIO object
    from io import BytesIO
    import base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Step 9: Convert image to base64 string
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 10: Prepare the context with the image
    context = {
        'graph_image': img_base64  # Pass the base64 image to the template
    }

    # Step 11: Render the template
    return render(request, 'dashboard/plot_ratings_trend.html', context)
