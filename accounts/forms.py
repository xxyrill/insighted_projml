from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser  # Ensure this is your custom user model

class CustomUserCreationForm(UserCreationForm):
    password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput,
        help_text='Enter a password',
    )
    password1 = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput,
        help_text='Enter the same password as before',
    )

    class Meta:
        model = CustomUser
        fields = ('username', 'password', 'password1', 'user_type')

    def clean_password2(self):
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords do not match.")
        return password2
