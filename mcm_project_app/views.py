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
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud



def login_page(request):
    return render(request, 'authentications/login.html')

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

    # Step 6: Group by 'deptname' and calculate the average total rating
    avg_ratings = df.groupby('dept_name')['total_rating'].mean().reset_index()

    # Step 7: Plot the average ratings for each department as a bar chart
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.bar(avg_ratings['dept_name'], avg_ratings['total_rating'], color='skyblue')
    plt.title('Department-wise Average Ratings')
    plt.xlabel('Department')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45, ha="right")
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

    # Step 10: Prepare the context with the image
    context = {
        'graph_image': img_base64  # Pass the base64 image to the template
    }

    # Step 11: Render the template
    return render(request, 'dashboard/plot_department_average_ratings.html', context)

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

    # Step 6: Plot the distribution using a histogram or box plot
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Option 1: Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings_data, bins=30, kde=True, color='blue')  # KDE for smooth distribution curve
    plt.title('Distribution of Ratings (Histogram)')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')

    # Option 2: Uncomment the following for box plot
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x=ratings_data, color='skyblue')
    # plt.title('Distribution of Ratings (Box Plot)')
    # plt.xlabel('Rating')

    # Step 7: Save the plot to a BytesIO object
    from io import BytesIO
    import base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Step 8: Convert image to base64 string
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 9: Prepare the context with the image
    context = {
        'graph_image': img_base64  # Pass the base64 image to the template
    }

    # Step 10: Render the template
    return render(request, 'dashboard/plot_rating_distribution.html', context)


def plot_word_clouds(request):
    # Step 1: Query the UploadCSV table to get comments data
    data = UploadCSV.objects.all().values('question_31', 'question_32')

    # Step 2: Convert the QuerySet to a DataFrame for easier manipulation
    import pandas as pd
    df = pd.DataFrame(list(data))

    # Step 3: Ensure the dataframe is not empty
    if df.empty:
        return HttpResponse("No data available to generate word clouds.", status=400)

    # Debugging: Print the DataFrame
    print("DataFrame contents:", df)

    # Step 4: Extract comments for instructors and courses, and combine them into single strings
    instructor_comments = " ".join(df['question_31'].dropna().tolist())
    course_comments = " ".join(df['question_32'].dropna().tolist())

    # Filter out unwanted comments
    instructor_comments = " ".join([comment for comment in instructor_comments.split() if comment.lower() not in ['none', 'n/a', 'no comment']])
    course_comments = " ".join([comment for comment in course_comments.split() if comment.lower() not in ['none', 'n/a', 'no comment']])

    # Debugging: Print combined comments
    print(f"Instructor Comments: '{instructor_comments}'")
    print(f"Course Comments: '{course_comments}'")

    # Step 5: Preprocess the comments by converting them to lowercase and removing common stop words
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from nltk.corpus import stopwords
    import string

    # Get stopwords and add punctuation to the stopwords list
    stop_words = set(stopwords.words('english') + list(string.punctuation))

    # Helper function to create word cloud
    def generate_word_cloud(text, title):
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              stopwords=stop_words, collocations=False).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        plt.tight_layout()

        # Save the plot to a BytesIO object
        from io import BytesIO
        import base64
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 6: Generate word clouds for both sets of comments
    instructor_wordcloud_base64 = generate_word_cloud(instructor_comments, 'Word Cloud for Instructor Comments')
    course_wordcloud_base64 = generate_word_cloud(course_comments, 'Word Cloud for Course Comments')

    # Debugging: Print word cloud data
    print("Instructor WordCloud Base64: ", instructor_wordcloud_base64)
    print("Course WordCloud Base64: ", course_wordcloud_base64)

    # Step 7: Prepare the context with both images
    context = {
        'instructor_wordcloud': instructor_wordcloud_base64,
        'course_wordcloud': course_wordcloud_base64
    }

    # Step 8: Render the template
    return render(request, 'dashboard/plot_word_clouds.html', context)

# FIX NO RATING
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
    sns.histplot(df['instructor_comment_length'], bins=30, color='blue', kde=True, label='Instructor Comments', alpha=0.5)
    sns.histplot(df['course_comment_length'], bins=30, color='orange', kde=True, label='Course Comments', alpha=0.5)

    plt.title('Distribution of Comment Lengths')
    plt.xlabel('Comment Length (characters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y')  # Optional: Add a grid for better readability
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Convert image to base64 string
    histogram_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 6: Prepare the context with the image
    context = {
        'histogram': histogram_base64
    }

    # Step 7: Render the template
    return render(request, 'dashboard/plot_length_of_comments_analysis.html', context)

# NO RATINGS / CANNOT BE PLOT
def plot_ratings_by_course(request):
    # Step 1: Query the UploadCSV table to get the required data (course names and questions 1 to 32)
    data = UploadCSV.objects.all().values('crs_name', 'question_1', 'question_2', 'question_3', 'question_4',
                                          'question_5', 'question_6', 'question_7', 'question_8', 'question_9',
                                          'question_10', 'question_11', 'question_12', 'question_13', 'question_14',
                                          'question_15', 'question_16', 'question_17', 'question_18', 'question_19',
                                          'question_20', 'question_21', 'question_22', 'question_23', 'question_24',
                                          'question_25', 'question_26', 'question_27', 'question_28', 'question_29',
                                          'question_30', 'question_31', 'question_32')

    # Step 2: Convert the QuerySet to a DataFrame for easier manipulation
    df = pd.DataFrame(list(data))

    # Step 3: Ensure the dataframe is not empty
    if df.empty:
        return HttpResponse("No data available to analyze questions.", status=400)

    # Step 4: Convert question columns to numeric, forcing non-numeric values to NaN
    for col in ['question_1', 'question_2', 'question_3', 'question_4', 'question_5', 'question_6', 'question_7', 'question_8', 'question_9',
                'question_10', 'question_11', 'question_12', 'question_13', 'question_14', 'question_15',]:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, forcing errors to NaN

    # Step 5: Reshape the DataFrame to have one column for the questions and their values
    df_melted = df.melt(id_vars=['crs_name'], value_vars=['question_1', 'question_2', 'question_3', 'question_4',
                                                          'question_5', 'question_6', 'question_7', 'question_8',
                                                          'question_9', 'question_10', 'question_11', 'question_12',
                                                          'question_13', 'question_14', 'question_15',],
                           var_name='question', value_name='rating')

    # Step 6: Calculate average ratings per course and per question
    average_ratings = df_melted.groupby(['crs_name', 'question'])['rating'].mean().reset_index()

    # Step 7: Generate the bar chart
    plt.figure(figsize=(20, 15))
    sns.barplot(x='crs_name', y='rating', hue='question', data=average_ratings, palette='viridis')
    plt.title('Average Responses for Questions 1-32 by Course')
    plt.xlabel('Course Name')
    plt.ylabel('Average Response')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Question')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Convert image to base64 string
    bar_chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 8: Prepare the context with the image
    context = {
        'bar_chart': bar_chart_base64
    }

    # Step 9: Render the template
    return render(request, 'dashboard/plot_ratings_by_course.html', context)

def plot_correlation_heatmap(request):
    # Step 1: Query the UploadCSV table to get necessary data
    data = UploadCSV.objects.all().values(
        'question_1', 'question_2', 'question_3', 'question_4', 'question_5',
        'question_6', 'question_7', 'question_8', 'question_9', 'question_10',
        'question_11', 'question_12', 'question_13', 'question_14', 'question_15',
        'question_16', 'question_17', 'question_18', 'question_19', 'question_20',
        'question_21', 'question_22', 'question_23', 'question_24', 'question_25',
        'question_26', 'question_27', 'question_28', 'question_29', 'question_30',
    )

    # Step 2: Convert the QuerySet to a DataFrame for easier manipulation
    df = pd.DataFrame(list(data))

    # Step 3: Ensure the dataframe is not empty
    if df.empty:
        return HttpResponse("No data available to generate heatmap.", status=400)

    # Step 4: Convert the data to numeric, coercing errors (like 'N/A') to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Step 5: Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Step 6: Generate the heatmap with a red-yellow-green color scheme
    plt.figure(figsize=(20, 15))  # Adjust the size as needed

    # Create a custom red-yellow-green colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ['red', 'yellow', 'green'])

    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt='.2f', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Convert image to base64 string
    heatmap_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 7: Prepare the context with the heatmap image
    context = {
        'heatmap': heatmap_base64
    }

    # Step 8: Render the template
    return render(request, 'dashboard/plot_correlation_heatmap.html', context)

# FIX NO RATING COLUMN
def plot_performance_by_year_and_department(request):
    # Step 1: Query the UploadCSV table to get necessary data
    data = UploadCSV.objects.all().values('crs_year', 'dept_name', 'rating')

    # Step 2: Convert the QuerySet to a DataFrame for easier manipulation
    df = pd.DataFrame(list(data))

    # Step 3: Ensure the dataframe is not empty
    if df.empty:
        return HttpResponse("No data available to generate the performance chart.", status=400)

    # Step 4: Calculate average ratings grouped by year and department
    avg_ratings = df.groupby(['crs_year', 'dept_name'])[
        ['question_1', 'question_2', 'question_3', 'question_4', 'question_5',
         'question_6', 'question_7', 'question_8', 'question_9', 'question_10',
         'question_11', 'question_12', 'question_13', 'question_14', 'question_15',
         'question_16', 'question_17', 'question_18', 'question_19', 'question_20',
         'question_21', 'question_22', 'question_23', 'question_24', 'question_25',
         'question_26', 'question_27', 'question_28', 'question_29', 'question_30',
         ]
    ].mean().reset_index()

    # Step 5: Pivot the DataFrame for plotting
    questions = [
        'question_1', 'question_2', 'question_3', 'question_4', 'question_5',
        'question_6', 'question_7', 'question_8', 'question_9', 'question_10',
        'question_11', 'question_12', 'question_13', 'question_14', 'question_15',
        'question_16', 'question_17', 'question_18', 'question_19', 'question_20',
        'question_21', 'question_22', 'question_23', 'question_24', 'question_25',
        'question_26', 'question_27', 'question_28', 'question_29', 'question_30',

    ]

    # Dictionary to store pivot tables for each question
    pivot_tables = {}

    for question in questions:
        pivot_tables[question] = avg_ratings.pivot(index='crs_year', columns='dept_name', values=question)

    # Step 6: Generate the bar chart
    plt.figure(figsize=(10, 6))
    pivot_df.plot(kind='bar', stacked=False)  # Change to `stacked=True` for a stacked bar chart
    plt.title('Average Ratings by Year and Department')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Convert image to base64 string
    performance_chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 7: Prepare the context with the chart image
    context = {
        'performance_chart': performance_chart_base64
    }

    # Step 8: Render the template
    return render(request, 'dashboard/performance_by_year_and_department.html', context)

# HOW TO GET THE SENTIMENT SCORE?
def plot_sentiment_analysis_over_time(request):
    # Step 1: Query the UploadCSV table to get necessary data
    # Assuming you have a sentiment_score column
    data = UploadCSV.objects.all().values('crs_year', 'sentiment_score')  # Include the sentiment score column

    # Step 2: Convert the QuerySet to a DataFrame for easier manipulation
    df = pd.DataFrame(list(data))

    # Step 3: Ensure the dataframe is not empty
    if df.empty:
        return HttpResponse("No data available to generate the sentiment analysis chart.", status=400)

    # Step 4: Calculate the average sentiment score grouped by year
    avg_sentiment = df.groupby('crs_year')['sentiment_score'].mean().reset_index()

    # Step 5: Generate the line chart
    plt.figure(figsize=(12, 6))
    plt.plot(avg_sentiment['crs_year'], avg_sentiment['sentiment_score'], marker='o', color='blue')
    plt.title('Average Sentiment Score Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Convert image to base64 string
    sentiment_chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Step 6: Prepare the context with the chart image
    context = {
        'sentiment_chart': sentiment_chart_base64
    }

    # Step 7: Render the template
    return render(request, 'dashboard/sentiment_analysis_over_time.html', context)

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
    return render(request, 'dashboard/plot_sentiment_analysis_over_time.html', context)


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
    return render(request, 'dashboard/plot_comparison_of_ratings_and_comment_length.html', context)

# 13. Instructor Comparison Dashboard

