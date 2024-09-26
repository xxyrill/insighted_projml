import pandas as pd
from django.db import connection
from django.conf import settings

def upload_csv_to_db(csv_file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file_path)

    # Convert DataFrame to a list of tuples containing the data
    data_tuples = [tuple(x) for x in data.values]

    # Define your insert SQL query (make sure to adjust the table and column names)
    # Replace 'accounts_uploadcsv' with your actual table name
    sql = "INSERT INTO accounts_uploadcsv (column1, column2, ...) VALUES %s"

    # Using psycopg2.extras to perform bulk insert
    with connection.cursor() as cursor:
        from psycopg2.extras import execute_values
        execute_values(cursor, sql, data_tuples)