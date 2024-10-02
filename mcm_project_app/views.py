import matplotlib
import nltk
matplotlib.use('Agg')  # Set the backend to Agg for non-interactive use
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from django.http import HttpResponse
from django.conf import settings
from .utilities import csvPathFileName
from django.shortcuts import render
import base64
from .utils import upload_csv_to_db
from django.contrib import messages
from io import BytesIO
from nltk.sentiment import SentimentIntensityAnalyzer
from django.db import IntegrityError
from nltk.corpus import stopwords
from django.http import JsonResponse
from accounts.models import UploadCSV
import re
from collections import Counter

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

    # Step 2: Insert new data into the UploadCSV table
    for _, row in df.iterrows():
        try:
            UploadCSV.objects.create(
                dept_name=row['deptname'],
                resp_fac=row['resp_fac'],
                crs_name=row['crs_name']
            )
        except IntegrityError as e:
            print(f"Error inserting row: {e}")  # Log or handle the error if necessary

    # Step 3: Extract unique values for filters
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

#1. Trend of Average Ratings per Term
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
    return JsonResponse({'image': img_base64})

# 2. Department-wise Average Ratings
def plot_department_average_ratings(request):
    # Step 1: Query the UploadCSV table to get the required data including department and ratings
    data = UploadCSV.objects.all().values('dept_name', 'question_1', 'question_2', 'question_3', 'question_4',
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
                        'question_31', 'question_32']

    df[question_columns] = df[question_columns].apply(pd.to_numeric, errors='coerce')
    df[question_columns] = df[question_columns].fillna(0)

    # Step 5: Calculate the total rating per row by summing the question columns
    df['total_rating'] = df[question_columns].sum(axis=1)

    # Step 6: Group by 'dept_name' and calculate the average total rating
    avg_ratings = df.groupby('dept_name')['total_rating'].mean().reset_index()

    # Step 7: Plot the average ratings for each department as a bar chart
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    rects = plt.bar(avg_ratings['dept_name'], avg_ratings['total_rating'], color='skyblue')

    # Add title and labels
    plt.title('Department-wise Average Ratings')
    plt.xlabel('Department')
    plt.ylabel('Average Rating')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Enable gridlines for the y-axis
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add numbers (height of each bar) on top of the bars
    for rect in rects:
        yval = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', fontsize=10)

    # Ensure the layout is adjusted properly
    plt.tight_layout()

    # Step 8: Save the plot to a BytesIO object
    from io import BytesIO
    import base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Step 9: Convert image to base64 string
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return JsonResponse({'image': img_base64})

#3. Distribution of Ratings
def plot_rating_distribution(request):
    # Step 1: Query the UploadCSV table to get the required data (all question ratings)
    data = UploadCSV.objects.all().values('question_1', 'question_2', 'question_3', 'question_4',
                                          'question_5', 'question_6', 'question_7', 'question_8',
                                          'question_9', 'question_10', 'question_11', 'question_12',
                                          'question_13', 'question_14', 'question_15', 'question_16',
                                          'question_17', 'question_18', 'question_19', 'question_20',
                                          'question_21', 'question_22', 'question_23', 'question_24',
                                          'question_25', 'question_26', 'question_27', 'question_28',
                                          'question_29', 'question_30', 'question_31', 'question_32')

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
                        'question_31', 'question_32']

    df[question_columns] = df[question_columns].apply(pd.to_numeric, errors='coerce')
    df[question_columns] = df[question_columns].fillna(0)

    # Step 5: Combine all question columns into a single series to analyze rating distribution
    ratings_data = df[question_columns].values.flatten()  # Flatten the DataFrame to 1D array

    # Step 6: Plot the distribution using a histogram
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(ratings_data, bins=30, kde=True, color='blue')  # KDE for smooth distribution curve
    plt.title('Distribution of Ratings (Histogram)')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')

    # Step 7: Add gridlines and display counts on top of bars
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add counts on top of bars
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:  # Check to avoid division by zero
            plt.text(patch.get_x() + patch.get_width() / 2, height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

    # Step 8: Save the plot to a BytesIO object
    from io import BytesIO
    import base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return JsonResponse({'image': img_base64})

#4. Pie Chart Comments
nltk.download('stopwords')
# Function to get word frequencies
def get_word_frequencies(comments):
    # Combine all comments into one text
    text = ' '.join(comments)

    # Clean the text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.lower().split()  # Convert to lowercase and split into words

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Count word frequencies
    word_counts = Counter(filtered_words)
    return word_counts.most_common(10)  # Get the top 10 words

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

# 9. Sentiment Analysis Over Time
def plot_sentiment_analysis_over_time(request):
    # Calculate sentiment scores
    df = calculate_sentiment_scores()

    # Ensure the dataframe is not empty
    if df.empty:
        return HttpResponse("No data available to generate the sentiment analysis chart.", status=400)

    # Step 4: Assuming you have a `crs_year` column to group by year
    # Add the year to your DataFrame
    df['crs_year'] = UploadCSV.objects.all().values_list('crs_year', flat=True)  # Adjust this line based on your model

    # Step 5: Calculate average sentiment score grouped by year
    avg_sentiment = df.groupby('crs_year').mean().reset_index()

    # Step 6: Generate the line chart
    plt.figure(figsize=(12, 6))
    plt.plot(avg_sentiment['crs_year'], avg_sentiment['sentiment_score_question_31'], marker='o', color='blue', label='Question 31 Sentiment')
    plt.plot(avg_sentiment['crs_year'], avg_sentiment['sentiment_score_question_32'], marker='o', color='orange', label='Question 32 Sentiment')
    plt.title('Average Sentiment Score Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Convert image to base64 string
    sentiment_chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 7: Prepare the context with the chart image
    context = {
        'sentiment_chart': sentiment_chart_base64
    }

    # Step 8: Render the template
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return JsonResponse({'image': img_base64})

# 12. Comparison of Ratings and Comment Length
def plot_comparison_of_ratings_and_comment_length(request):
    # Step 1: Query your data
    data = UploadCSV.objects.all().values(
        'question_1', 'question_2', 'question_3', 'question_4',
        'question_5', 'question_6', 'question_7', 'question_8',
        'question_9', 'question_10', 'question_11', 'question_12',
        'question_13', 'question_14', 'question_15', 'question_16',
        'question_17', 'question_18', 'question_19', 'question_20',
        'question_21', 'question_22', 'question_23', 'question_24',
        'question_25', 'question_26', 'question_27', 'question_28',
        'question_29', 'question_30', 'question_31', 'question_32'
    )  # Adjust fields as necessary
    df = pd.DataFrame(list(data))

    # Step 2: Ensure the dataframe is not empty
    if df.empty:
        return HttpResponse("No data available to generate the comparison chart.", status=400)

    # Step 3: Convert relevant columns to numeric for ratings
    rating_columns = [f'question_{i}' for i in range(1, 31)]  # List of question columns from question_1 to question_30
    df[rating_columns] = df[rating_columns].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, coercing errors

    # Optionally fill NaN values with 0
    df[rating_columns] = df[rating_columns].fillna(0)

    # Calculate the total rating for each row
    df['total_rating'] = df[rating_columns].sum(axis=1)  # Calculate the total rating for each row

    # Step 4: Calculate the length of comments (string values)
    df['instructor_comment_length'] = df['question_31'].astype(str).str.len()  # Length of instructor comments
    df['course_comment_length'] = df['question_32'].astype(str).str.len()  # Length of course comments

    # Step 5: Create a new DataFrame for plotting
    comparison_data = {
        'Comment Length': [],
        'Total Rating': [],
        'Comment Type': []
    }

    # Add instructor comments data
    comparison_data['Comment Length'] += df['instructor_comment_length'].tolist()
    comparison_data['Total Rating'] += df['total_rating'].tolist()
    comparison_data['Comment Type'] += ['Instructor'] * len(df)

    # Add course comments data
    comparison_data['Comment Length'] += df['course_comment_length'].tolist()
    comparison_data['Total Rating'] += df['total_rating'].tolist()  # Total rating should be repeated for course comments
    comparison_data['Comment Type'] += ['Course'] * len(df)

    comparison_df = pd.DataFrame(comparison_data)

    # Step 6: Create the scatter plot
    plt.figure(figsize=(18, 10))

    # Plot instructor comments
    plt.scatter(
        comparison_df[comparison_df['Comment Type'] == 'Instructor']['Comment Length'],
        comparison_df[comparison_df['Comment Type'] == 'Instructor']['Total Rating'],
        color='red', label='Instructor Comments', alpha=0.7, s=100, edgecolor='black'
    )

    # Plot course comments
    plt.scatter(
        comparison_df[comparison_df['Comment Type'] == 'Course']['Comment Length'],
        comparison_df[comparison_df['Comment Type'] == 'Course']['Total Rating'],
        color='blue', label='Course Comments', alpha=0.7, s=100, edgecolor='black'
    )

    # Step 7: Add titles and labels
    plt.title('Comparison of Ratings and Comment Length', fontsize=16)
    plt.xlabel('Comment Length (Words)', fontsize=14)
    plt.ylabel('Total Rating', fontsize=14)

    # Step 8: Add grid and legend
    plt.grid(True)
    plt.legend(title='Comment Types')

    # Step 9: Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()

    # Convert image to base64 string
    comparison_chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 10: Prepare the context with the chart image
    context = {
        'comparison_chart': comparison_chart_base64
    }

    # Step 11: Render the template
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return JsonResponse({'image': img_base64})

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

# 11. Pareto Analysis of Courses
def plot_pareto_analysis(request):
    # Step 1: Query course data and ratings from the database (adjust field names accordingly)
    data = UploadCSV.objects.all().values('crs_name', 'question_1', 'question_2', 'question_3', 'question_4')  # Modify questions as per your needs

    # Step 2: Convert the QuerySet to a DataFrame
    df = pd.DataFrame(list(data))

    # Step 3: Calculate average rating for each course
    df['avg_rating'] = df[['question_1', 'question_2', 'question_3', 'question_4']].mean(axis=1)

    # Step 4: Group by course and calculate average rating per course
    course_ratings = df.groupby('crs_name')['avg_rating'].mean().reset_index()

    # Step 5: Sort courses by average rating in descending order
    course_ratings = course_ratings.sort_values(by='avg_rating', ascending=False)

    # Step 6: Keep only the top N courses
    top_n = 10
    course_ratings = course_ratings.head(top_n)

    # Step 7: Calculate cumulative percentage contribution
    course_ratings['cumulative_percentage'] = course_ratings['avg_rating'].cumsum() / course_ratings['avg_rating'].sum() * 100

    # Step 8: Plot the Pareto chart (bar chart with cumulative line)
    fig, ax1 = plt.subplots(figsize=(14, 12))  # Increased figure size

    # Bar chart for course ratings
    bars = ax1.bar(course_ratings['crs_name'], course_ratings['avg_rating'], color='C0', width=0.6)  # Narrower bars
    ax1.set_xlabel('Courses')
    ax1.set_ylabel('Average Rating', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')

    # Rotate and adjust the position of x-axis labels
    plt.xticks(rotation=60, ha='right')

    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # Line chart for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(course_ratings['crs_name'], course_ratings['cumulative_percentage'], color='C1', marker='o', linestyle='-', linewidth=2)
    ax2.set_ylabel('Cumulative Percentage', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    # Add gridlines
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Pareto chart title
    plt.title('Pareto Analysis of Courses')

    # Step 9: Save the chart to a BytesIO object
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Step 10: Convert the image to base64 string
    pareto_chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 11: Pass the base64 image to the template
    return JsonResponse({'image': pareto_chart_base64})

