from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .forms import CustomUserCreationForm, DatEntryForm

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
        form = DatEntryForm(request.POST)
        if form.is_valid():
            form.save()
            # If using AJAX, return success status
            return JsonResponse({'status': 'success', 'message': 'Data uploaded successfully!'})
        else:
            # If form is invalid, return errors in JSON format
            return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    else:
        form = DatEntryForm()
    return render(request, 'dashboard/upload_data.html', {'form': form})
