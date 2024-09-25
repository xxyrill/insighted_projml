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

from django.shortcuts import render
from django.urls import reverse


def login_page(request):
    return render(request, 'authentications/login.html')


def plot_ratings_trend(request):
    # Extract filter values from GET request
    term = request.GET.get('term', 'All Terms')
    department = request.GET.get('department', 'All Departments')
    instructor = request.GET.get('instructor', 'Respective Facilitator')
    course = request.GET.get('course', 'All Courses')

    csv_file_path = csvPathFileName()

    # Load CSV file
    data = pd.read_csv(csv_file_path)

    # Apply filters based on the correct column indices
    if term != 'All Terms':
        data = data[data.iloc[:, 5] == term]  # Term is the 6th column (index 5)
    if department != 'All Departments':
        data = data[data.iloc[:, 6] == department]  # Department is the 7th column (index 6)
    if instructor != 'Respective Facilitator':
        data = data[data.iloc[:, 8] == instructor]  # Instructor is the 9th column (index 8)
    if course != 'All Courses':
        data = data[data.iloc[:, 4] == course]  # Course is the 5th column (index 4)

    # Ensure data is filtered correctly, and now proceed with graph generation
    columns = data.columns[20:49]  # Assuming rating columns are from index 20 to 48
    data = data.dropna(subset=columns)  # Drop rows where ratings are NaN
    data['total_rating'] = data[columns].sum(axis=1)  # Summing across all rating columns

    # Group by 'crsyear' (assuming this column stores the term/year)
    avg_ratings = data.groupby('crsyear')['total_rating'].mean().reset_index()

    # Plotting the trend graph
    plt.figure(figsize=(10, 6))
    plt.plot(avg_ratings['crsyear'], avg_ratings['total_rating'], marker='o', linestyle='-', color='b')
    plt.title('Trend of Average Ratings Over Terms')
    plt.xlabel('Academic Term')
    plt.ylabel('Average Rating')
    plt.grid(True)

    # Save the plot to an in-memory file and return it as an HTTP response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return HttpResponse(img, content_type='image/png')


def plot_department_avg_ratings(request):
    csv_file_path = csvPathFileName()

    data = pd.read_csv(csv_file_path)

    columns = data.columns[20:49]

    data = data.dropna(subset=columns)

    data['total_rating'] = data[columns].sum(axis=1)

    avg_ratings = data.groupby(['deptname', 'crsyear'])['total_rating'].mean().unstack()

    plt.figure(figsize=(12, 8))

    sns.heatmap(avg_ratings, annot=True, cmap='RdYlGn', fmt='.1f')

    plt.title('Department-wise Average Ratings')
    plt.xlabel('Academic Term')
    plt.ylabel('Department')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plt.close()

    return HttpResponse(img, content_type='image/png')


def plot_rating_distribution(request):
    csv_file_path = csvPathFileName()

    data = pd.read_csv(csv_file_path)

    columns = data.columns[20:49]

    data = data.dropna(subset=columns)

    data['total_rating'] = data[columns].sum(axis=1)

    plt.figure(figsize=(12, 8))

    plt.hist(data['total_rating'], bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Ratings')
    plt.xlabel('Total Rating')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Box Plot
    # sns.boxplot(y=data['total_rating'])
    # plt.title('Box Plot of Ratings')
    # plt.ylabel('Total Rating')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Close the plot to free up memory
    plt.close()

    # Return the plot as a response
    return HttpResponse(img, content_type='image/png')


def generate_word_cloud(text):
    # Check if text is not empty
    if len(text.strip()) > 0:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        return wordcloud
    return None


def plot_word_clouds(request):
    csv_file_path = csvPathFileName()

    data = pd.read_csv(csv_file_path)

    if 'Comments on the Instructor' not in data.columns or 'Comments on the Course' not in data.columns:
        return HttpResponse("Columns 'AY' or 'AZ' not found in CSV.", status=400)

    comments_instructor = data['Comments on the Instructor'].astype(str).str.cat(sep=' ')
    comments_course = data['Comments on the Course'].astype(str).str.cat(sep=' ')

    wordcloud_instructor = generate_word_cloud(comments_instructor)
    wordcloud_course = generate_word_cloud(comments_course)

    plt.figure(figsize=(14, 8))

    if wordcloud_instructor:
        plt.subplot(1, 2, 1)
        plt.imshow(wordcloud_instructor, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Instructor Comments')
    else:
        plt.subplot(1, 2, 1)
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
        plt.axis('off')
        plt.title('Word Cloud - Instructor Comments')

    if wordcloud_course:
        plt.subplot(1, 2, 2)
        plt.imshow(wordcloud_course, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Course Comments')
    else:
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
        plt.axis('off')
        plt.title('Word Cloud - Course Comments')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plt.close()

    return HttpResponse(img, content_type='image/png')


def plot_comment_lengths(request):
    csv_file_path = csvPathFileName()

    data = pd.read_csv(csv_file_path)

    if 'Comments on the Instructor' not in data.columns or 'Comments on the Course' not in data.columns:
        return HttpResponse("Columns 'AY' or 'AZ' not found in CSV.", status=400)
    data['instructor_comment_length'] = data['Comments on the Instructor'].astype(str).apply(len)
    data['course_comment_length'] = data['Comments on the Course'].astype(str).apply(len)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(data['instructor_comment_length'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Instructor Comment Lengths')
    plt.xlabel('Length of Comments')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(data['course_comment_length'], bins=30, color='salmon', edgecolor='black')
    plt.title('Distribution of Course Comment Lengths')
    plt.xlabel('Length of Comments')
    plt.ylabel('Frequency')
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plt.close()

    return HttpResponse(img, content_type='image/png')


def plot_ratings_by_course(request):
    # Path to your CSV file
    csv_file_path = csvPathFileName()
    # Define the index range for the rating columns
    data = pd.read_csv(csv_file_path, header=0)
    rating_columns = data.columns[20:49]  # Adjust these indices based on your needs

    # Ensure necessary columns exist
    if 'crsname' not in data.columns or rating_columns.empty:
        return HttpResponse("Necessary columns not found in CSV.", status=400)

    # Drop rows where rating columns have NaN values
    data = data.dropna(subset=rating_columns)

    # Calculate the total rating for each row by summing across the rating columns
    data['total_rating'] = data[rating_columns].sum(axis=1)

    # Create a DataFrame with course names and their corresponding total ratings
    course_ratings = pd.DataFrame({
        'crsname': data['crsname'],
        'total_rating': data['total_rating']
    })

    # Calculate the average rating for each course
    avg_ratings_per_course = course_ratings.groupby('crsname')['total_rating'].mean().reset_index()

    # Ensure the DataFrame is not empty
    if avg_ratings_per_course.empty:
        return HttpResponse("No ratings data available.", status=400)

    # Create a bar chart for average ratings by course
    plt.figure(figsize=(12, 8))
    plt.bar(avg_ratings_per_course['crsname'], avg_ratings_per_course['total_rating'], color='lightblue',
            edgecolor='black')
    plt.xlabel('Course Name')
    plt.ylabel('Total Rating')
    plt.title('Total Ratings by Course')
    plt.xticks(rotation=90)  # Rotate course names for better readability
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Close the plot to free up memory
    plt.close()

    # Return the plot as a response
    return HttpResponse(img, content_type='image/png')


def plot_correlation_heatmap(request):
    # Path to your CSV file
    csv_file_path = csvPathFileName()

    # Read the CSV file with the first row as headers
    data = pd.read_csv(csv_file_path, header=0)

    # Define the rating columns by their index range
    rating_columns = data.columns[20:49]  # Adjust these indices as needed

    # Drop rows where rating columns have NaN values
    data = data.dropna(subset=rating_columns)

    # Calculate the total rating for each row by summing across the rating columns
    data['total_rating'] = data[rating_columns].sum(axis=1)

    # Extract comment columns
    data['instructor_comment_length'] = data['Comments on the Instructor'].astype(str).apply(len)
    data['course_comment_length'] = data['Comments on the Course'].astype(str).apply(len)

    # Select relevant numerical columns for correlation
    correlation_columns = ['total_rating', 'instructor_comment_length', 'course_comment_length']

    # Ensure the DataFrame contains the relevant columns
    if not all(col in data.columns for col in correlation_columns):
        return HttpResponse("Necessary columns not found in CSV.", status=400)

    # Compute the correlation matrix
    correlation_matrix = data[correlation_columns].corr()

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Ratings and Comment Lengths')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Close the plot to free up memory
    plt.close()

    # Return the plot as a response
    return HttpResponse(img, content_type='image/png')


def plot_performance_by_year_and_department(request):
    # Path to your CSV file
    csv_file_path = csvPathFileName()

    # Read the CSV file with the first row as headers
    data = pd.read_csv(csv_file_path, header=0)

    # Define the rating columns by their index range
    rating_columns = data.columns[20:49]  # Adjust these indices as needed

    # Drop rows where rating columns have NaN values
    data = data.dropna(subset=rating_columns)

    # Calculate the total rating for each row by summing across the rating columns
    data['total_rating'] = data[rating_columns].sum(axis=1)

    # Group by 'crsyear' and 'deptname' and calculate average rating
    performance_by_year_dept = data.groupby(['crsyear', 'deptname'])['total_rating'].mean().reset_index()

    # Pivot the DataFrame for better plotting
    pivot_table = performance_by_year_dept.pivot(index='crsyear', columns='deptname', values='total_rating')

    # Ensure the DataFrame is not empty
    if pivot_table.empty:
        return HttpResponse("No performance data available.", status=400)

    # Create a stacked bar chart
    plt.figure(figsize=(14, 8))
    pivot_table.plot(kind='bar', stacked=True, colormap='tab20', ax=plt.gca())
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.title('Average Ratings by Year and Department')
    plt.legend(title='Department')
    plt.xticks(rotation=45)  # Rotate year labels for better readability
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Close the plot to free up memory
    plt.close()

    # Return the plot as a response
    return HttpResponse(img, content_type='image/png')


def plot_sentiment_over_time(request):
    # Path to your CSV file
    csv_file_path = csvPathFileName()

    # Read the CSV file with the first row as headers
    data = pd.read_csv(csv_file_path, header=0)

    # Define the columns for comments
    comments_instructor_col = 'Comments on the Instructor'
    comments_course_col = 'Comments on the Course'

    # Perform sentiment analysis
    def get_sentiment(text):
        # Avoid errors with empty or non-string data
        if pd.isna(text) or not isinstance(text, str):
            return 0
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    # Apply sentiment analysis
    data['instructor_sentiment'] = data[comments_instructor_col].apply(get_sentiment)
    data['course_sentiment'] = data[comments_course_col].apply(get_sentiment)

    # Add a term column based on 'crsyear' if it is available
    if 'crsyear' not in data.columns:
        return HttpResponse("Column 'crsyear' not found in CSV.", status=400)

    # Calculate average sentiment per term (crsyear)
    sentiment_by_term = data.groupby('crsyear')[['instructor_sentiment', 'course_sentiment']].mean().reset_index()

    # Plot the average sentiment scores over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=sentiment_by_term, x='crsyear', y='instructor_sentiment', label='Instructor Comments Sentiment',
                 marker='o')
    sns.lineplot(data=sentiment_by_term, x='crsyear', y='course_sentiment', label='Course Comments Sentiment',
                 marker='o')
    plt.xlabel('Academic Year')
    plt.ylabel('Average Sentiment Score')
    plt.title('Sentiment Analysis Over Time')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Close the plot to free up memory
    plt.close()

    # Return the plot as a response
    return HttpResponse(img, content_type='image/png')


def plot_instructor_performance_over_time(request):
    # Path to your CSV file
    csv_file_path = csvPathFileName()

    # Read the CSV file with the first row as headers
    data = pd.read_csv(csv_file_path, header=0)

    # Define the columns for ratings and instructors
    rating_columns = data.columns[20:49]  # Adjust these indices as needed
    instructor_col = 'resp_fac'  # Ensure this matches the column name for instructor names
    term_col = 'crsyear'  # Ensure this matches the column name for terms

    # Drop rows with NaN values in the rating columns
    data = data.dropna(subset=rating_columns)

    # Calculate total rating for each row
    data['total_rating'] = data[rating_columns].sum(axis=1)

    # Group by 'crsyear' and 'facultyname' and calculate average rating
    performance_by_instructor = data.groupby([term_col, instructor_col])['total_rating'].mean().reset_index()

    # Pivot the DataFrame for plotting
    pivot_table = performance_by_instructor.pivot(index=term_col, columns=instructor_col, values='total_rating')

    # Ensure the DataFrame is not empty
    if pivot_table.empty:
        return HttpResponse("No performance data available.", status=400)

    # Create a line chart with multiple lines
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=performance_by_instructor, x=term_col, y='total_rating', hue=instructor_col, marker='o')
    plt.xlabel('Academic Year')
    plt.ylabel('Average Rating')
    plt.title('Instructor Performance Over Time')
    plt.legend(title='Instructor Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)

    # Close the plot to free up memory
    plt.close()

    # Return the plot as a response
    return HttpResponse(img, content_type='image/png')


def plot_pareto_analysis(request):
    # Path to your CSV file
    csv_file_path = csvPathFileName()

    # Read the CSV file with the first row as headers
    data = pd.read_csv(csv_file_path, header=0)

    # Define the columns for ratings and courses
    rating_columns = data.columns[20:49]  # Adjust these indices as needed
    course_col = 'crsname'  # Ensure this matches the column name for course names

    # Drop rows with NaN values in the rating columns
    data = data.dropna(subset=rating_columns)

    # Calculate total rating for each row
    data['total_rating'] = data[rating_columns].sum(axis=1)

    # Calculate average rating for each course
    average_ratings = data.groupby(course_col)['total_rating'].mean().reset_index()

    # Rank courses by average rating
    average_ratings = average_ratings.sort_values(by='total_rating', ascending=False)

    # Calculate cumulative contribution
    average_ratings['cumulative_percentage'] = average_ratings['total_rating'].cumsum() / average_ratings[
        'total_rating'].sum() * 100

    # Create a Pareto chart
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Bar chart for average ratings
    sns.barplot(x=course_col, y='total_rating', data=average_ratings, ax=ax1, color='b')
    ax1.set_xlabel('Course Name')
    ax1.set_ylabel('Average Rating', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Line chart for cumulative percentage
    ax2 = ax1.twinx()
    sns.lineplot(x=average_ratings[course_col], y='cumulative_percentage', data=average_ratings, ax=ax2, color='r',
                 marker='o', linestyle='--')
    ax2.set_ylabel('Cumulative Percentage', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Pareto Analysis of Courses')
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)

    # Close the plot to free up memory
    plt.close()

    # Return the plot as a response
    return HttpResponse(img, content_type='image/png')


def plot_ratings_vs_comment_length(request):
    # Path to your CSV file
    csv_file_path = csvPathFileName()

    # Read the CSV file with the first row as headers
    data = pd.read_csv(csv_file_path, header=0)

    # Define the columns for ratings, comment lengths, and comments
    rating_columns = data.columns[20:49]  # Adjust these indices as needed
    instructor_comment_col = 'Comments on the Instructor'  # Column with comments on the instructor
    course_comment_col = 'Comments on the Course'  # Column with comments on the course

    # Drop rows with NaN values in the rating columns
    data = data.dropna(subset=rating_columns)

    # Calculate total rating for each row
    data['total_rating'] = data[rating_columns].sum(axis=1)

    # Calculate the length of comments
    data['instructor_comment_length'] = data[instructor_comment_col].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0)
    data['course_comment_length'] = data[course_comment_col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    # Create a scatter plot comparing ratings with instructor comment length
    fig, ax1 = plt.subplots(figsize=(14, 8))

    sns.scatterplot(x='instructor_comment_length', y='total_rating', data=data, ax=ax1, color='b',
                    label='Instructor Comments')
    ax1.set_xlabel('Instructor Comment Length (Words)')
    ax1.set_ylabel('Total Rating', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    plt.grid(True)

    # Optionally, you can add a second scatter plot for course comments
    ax2 = ax1.twinx()
    sns.scatterplot(x='course_comment_length', y='total_rating', data=data, ax=ax2, color='r', label='Course Comments',
                    marker='o')
    ax2.set_ylabel('Total Rating', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.grid(True)

    # Add titles and legends
    plt.title('Comparison of Ratings and Comment Length')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)

    # Close the plot to free up memory
    plt.close()

    # Return the plot as a response
    return HttpResponse(img, content_type='image/png')


def plot_instructor_comparison_dashboard(request):
    # Path to your CSV file
    csv_file_path = csvPathFileName()

    # Read the CSV file with the first row as headers
    data = pd.read_csv(csv_file_path, header=0)

    # Define columns and clean data
    rating_columns = data.columns[20:49]  # Adjust these indices as needed
    instructor_comment_col = 'Comments on the Instructor'  # Column with comments on the instructor
    course_comment_col = 'Comments on the Course'  # Column with comments on the course
    instructor_col = 'resp_fac'  # Column with instructor names
    term_col = 'crsyear'  # Column with terms

    # Drop rows with NaN values in the rating columns
    data = data.dropna(subset=rating_columns)

    # Calculate total rating for each row
    data['total_rating'] = data[rating_columns].sum(axis=1)

    # Calculate average rating per instructor
    instructor_avg_ratings = data.groupby(instructor_col)['total_rating'].mean().reset_index()
    instructor_avg_ratings = instructor_avg_ratings.sort_values(by='total_rating', ascending=False)

    # Create line chart for instructor performance over time
    fig, ax1 = plt.subplots(figsize=(14, 8))
    for instructor in data[instructor_col].unique():
        instructor_data = data[data[instructor_col] == instructor]
        sns.lineplot(x=term_col, y='total_rating', data=instructor_data, ax=ax1, label=instructor)
    ax1.set_xlabel('Term')
    ax1.set_ylabel('Total Rating')
    ax1.set_title('Instructor Performance Over Time')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Save the line chart to a BytesIO object
    line_chart_img = io.BytesIO()
    plt.savefig(line_chart_img, format='png', bbox_inches='tight')
    line_chart_img.seek(0)
    plt.close()

    # Create bar chart for average ratings by instructor
    fig, ax2 = plt.subplots(figsize=(14, 8))
    sns.barplot(x=instructor_col, y='total_rating', data=instructor_avg_ratings, ax=ax2, palette='viridis')
    ax2.set_xlabel('Instructor')
    ax2.set_ylabel('Average Rating')
    ax2.set_title('Average Ratings by Instructor')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.grid(True)

    # Save the bar chart to a BytesIO object
    bar_chart_img = io.BytesIO()
    plt.savefig(bar_chart_img, format='png', bbox_inches='tight')
    bar_chart_img.seek(0)
    plt.close()

    # Generate word cloud for instructor comments
    comments = data[instructor_comment_col].dropna().astype(str).tolist()
    text = ' '.join(comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Save the word cloud to a BytesIO object
    fig, ax3 = plt.subplots(figsize=(12, 6))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis('off')
    ax3.set_title('Word Cloud of Instructor Comments')

    wordcloud_img = io.BytesIO()
    plt.savefig(wordcloud_img, format='png', bbox_inches='tight')
    wordcloud_img.seek(0)
    plt.close()

    # Combine the images into a single dashboard
    from PIL import Image

    # Load images
    line_chart = Image.open(line_chart_img)
    bar_chart = Image.open(bar_chart_img)
    wordcloud = Image.open(wordcloud_img)

    # Create a new image to combine the plots
    combined_img = Image.new('RGB', (line_chart.width, line_chart.height + bar_chart.height + wordcloud.height),
                             'white')

    # Paste images into the combined image
    combined_img.paste(line_chart, (0, 0))
    combined_img.paste(bar_chart, (0, line_chart.height))
    combined_img.paste(wordcloud, (0, line_chart.height + bar_chart.height))

    # Save the combined image to a BytesIO object
    combined_img_io = io.BytesIO()
    combined_img.save(combined_img_io, format='png')
    combined_img_io.seek(0)

    # Return the combined image as a response
    return HttpResponse(combined_img_io, content_type='image/png')