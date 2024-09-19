from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .forms import CustomUserCreationForm
from .models import DatEntry
from django.utils import timezone

def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
        else:
            print(form.errors)
    else:
        form = CustomUserCreationForm()
    return render(request, 'authentications/registration.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('home')
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
    return render(request, 'authentications/login.html')

@login_required
def upload_data(request):
    if request.method == 'POST':
        survey_name = request.POST.get('survey_name')
        date_start = request.POST.get('date_start')
        date_close = request.POST.get('date_close')
        crs_name = request.POST.get('crs_name')
        crs_year = request.POST.get('crs_year')
        dept_name = request.POST.get('dept_name')
        crs_dir = request.POST.get('crs_dir')
        resp_fac = request.POST.get('resp_fac')
        eval_id = request.POST.get('eval_id')
        eval_uname = request.POST.get('eval_uname')
        eval_email = request.POST.get('eval_email')
        t_submit = timezone.now()  # Automatically set current time as submission time
        mobile = request.POST.get('mobile') == 'on'  # Checkbox, returns True if checked
        grad_year = request.POST.get('grad_year')
        gender = request.POST.get('gender')
        program = request.POST.get('program')
        research_1 = request.POST.get('research_1')
        research_2 = request.POST.get('research_2')
        research_3 = request.POST.get('research_3')
        question_1 = request.POST.get('question_1')

        # Validate and convert fields if needed, for example:
        try:
            date_start = timezone.datetime.strptime(date_start, '%Y-%m-%d').date()
            date_close = timezone.datetime.strptime(date_close, '%Y-%m-%d').date()
            grad_year = timezone.datetime.strptime(grad_year, '%Y-%m-%d').date()
            crs_year = int(crs_year)
            eval_id = int(eval_id)
            question_1 = int(question_1)
        except (ValueError, TypeError):
            # Handle invalid data or return an error
            return render(request, 'dashboard/upload_data.html', {'error': 'Invalid input'})

        # Save the data to the database
        DatEntry.objects.create(
            survey_name=survey_name,
            date_start=date_start,
            date_close=date_close,
            crs_name=crs_name,
            crs_year=crs_year,
            dept_name=dept_name,
            crs_dir=crs_dir,
            resp_fac=resp_fac,
            eval_id=eval_id,
            eval_uname=eval_uname,
            eval_email=eval_email,
            t_submit=t_submit,
            mobile=mobile,
            grad_year=grad_year,
            gender=gender,
            program=program,
            research_1=research_1,
            research_2=research_2,
            research_3=research_3,
            question_1=question_1
        )

        return redirect('success')  # Redirect to a success page after saving
    return render(request, 'dashboard/toaster.html')
