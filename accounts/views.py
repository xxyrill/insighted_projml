from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from .forms import CustomUserCreationForm

def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        # username = request.POST.get('username')
        # password = request.POST.get('password')

        if form.is_valid():
            form.save()
            return redirect('login')
            # print(f"User created: {user.username}")
            # user = authenticate(username=user.username, password=form.cleaned_data['password'])
            # # print("User authenticated successfully!")
            # if user is not None:
            #     # print("User authenticated successfully!")
            #     auth_login(request, user)
            #     return redirect('dashboard')
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
        # form = AuthenticationForm(request, data=request.POST)
        # if form.is_valid():
        #     user = form.get_user()
        #     auth_login(request, user)
        #     return redirect('dashboard')  # Redirect to the dashboard after login
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