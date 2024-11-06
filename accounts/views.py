import time
import pandas as pd
import numpy as np
import csv

from django import forms
from .forms import UserCreationForm
from django.shortcuts import render, redirect
from .models import CustomUser, UploadCSV
from .forms import CustomUserCreationForm
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.utils import timezone
from django.shortcuts import render, redirect
from .forms import CustomUserCreationForm  # Import your form from forms.py
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import CustomUserCreationForm  # Import your custom form
from django.dispatch import receiver
from django.db.models.signals import post_save
from django.contrib.auth.models import User


def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            # Use Django's messages framework to display a success message
            messages.success(request, 'Account created successfully!')
            return redirect('register')  # Redirect to the same page

    else:
        form = CustomUserCreationForm()

    return render(request, 'dashboard/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'authentications/login.html', {'form': form})

@login_required
def dashboard_view(request):
    user_type = request.user.user_type
    context = {
        'user_type': user_type
    }
    return render(request, 'dashboard/index.html', context)

@login_required
def dashboardgraph(request):
    user_type = request.user.user_type
    context = {
        'user_type': user_type
    }
    return render(request, 'dashboard/dashboard.html', context)

def logout_view(request):
    auth_logout(request)  # Logs the user out
    return render(request, 'authentications/login.html')

def about_us(request):
    user_type = request.user.user_type
    context = {
        'user_type': user_type
    }
    return render(request, 'dashboard/about_us.html',context)

@login_required
def create_account(request):
    return render(request, 'dashboard/create_account.html')

# Delete database
def delete_data(request):
    if request.method == 'POST':
        if request.user.is_authenticated:
            # Step 1: Clear existing data in the UploadCSV table
            UploadCSV.objects.all().delete()
            messages.success(request, "All data deleted successfully.")
        else:
            messages.error(request, "You are not authorized to delete data.")

        return redirect('dashboard')  # Redirect to your dashboard or wherever appropriate

def clean_question_data(df):
    # Loop through each column that represents a question
    question_columns = [col for col in df.columns if col.startswith('question_')]
    
    # Convert columns to numeric, coercing errors to NaN (keeping NaNs as they are for empty values)
    df[question_columns] = df[question_columns].apply(pd.to_numeric, errors='coerce')
    
    return df

@login_required
def upload_csv(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        # Read the CSV file into a DataFrame
        csv_file = request.FILES['csv_file']
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            messages.error(request, f'Error reading CSV file: {e}')
            return redirect('upload_csv')

        # Clean the question data
        df = clean_question_data(df)

        reader = df.itertuples(index=False, name=None)  # Use iterator to loop through rows
        total_rows = len(df)

        request.session['progress'] = 0  # Initialize progress

        try:
            for index, row in enumerate(reader):
                # Detect term based on survey_name
                survey_name = row[0]  # Assuming the survey name is the first column
                if '23.24.1T | College Faculty Performance Evaluation' in survey_name:
                    term = 'Term 1'
                elif '2T.23.24 | College Faculty Performance Evaluation' in survey_name:
                    term = 'Term 2'
                elif '3T.23.24 | College Faculty Evaluation Survey' in survey_name:
                    term = 'Term 3'
                else:
                    term = 'Unknown Term'  # Handle case where term is not recognized

                # Extract question values, setting them to actual CSV values or None if missing
                question_values = {
                    f'question_{i - 19}': row[i] if pd.notna(row[i]) else None
                    for i in range(20, 52)
                }

                try:
                    UploadCSV.objects.create(
                        survey_name=survey_name,
                        term=term,
                        date_start=row[1],
                        date_close=row[2],
                        crs_num=row[3],
                        crs_name=row[4],
                        crs_year=row[5],
                        dept_name=row[6],
                        crs_dir=row[7],
                        resp_fac=row[8],
                        eval_id=row[9],
                        eval_uname=row[10],
                        eval_email=row[11],
                        t_submit=row[12],
                        mobile=bool(int(row[13])),  # Convert 1/0 to boolean
                        grad_year=row[14],
                        gender=row[15],
                        program=row[16],
                        research_1=row[17],
                        research_2=row[18],
                        research_3=row[19],
                        # Pass the question values dynamically
                        **question_values
                    )
                except (IndexError, ValueError) as e:
                    print(f"Error in row processing: {e}, row: {row}")  # Log the row causing the issue
                    continue  # Skip this row and continue with the next

                # Update progress
                progress_percentage = (index + 1) / total_rows * 100
                request.session['progress'] = progress_percentage  # Update session with progress

            messages.success(request, 'CSV data has been successfully uploaded.')
            request.session['progress'] = 100  # Mark completion
            return redirect('dashboard')  # Redirect after processing all rows

        except Exception as e:
            messages.error(request, f'Error processing file: {e}')
            return redirect('upload_csv')

    # Handle GET request to render the upload form
    return render(request, 'dashboard/dashboard.html')

def check_progress(request):
    progress = request.session.get('progress', 0)
    return JsonResponse({'progress': progress})

def is_dean(user):
    return hasattr(user, 'profile') and user.profile.department in ['ATYCB', 'CAS', 'CEA', 'CCIS']

def view_department_data(request):
    if request.user.user_type in ['ATYCB', 'CAS', 'CEA', 'CCIS']:
        # Show only the department data related to the dean's user type
        department_data = UploadCSV.objects.filter(dept_name=request.user.user_type)
    else:
        # Handle other user types
        department_data = UploadCSV.objects.all()

# Render the data in a template
# return render(request, 'dashboard/department_data.html', {'data': data})