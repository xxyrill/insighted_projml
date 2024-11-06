from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser
from django.shortcuts import render, redirect
from django.contrib import messages


class CustomUserCreationForm(UserCreationForm):
    password1 = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Enter a secure password'}),
        help_text='Enter a password',
    )
    password2 = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Re-enter your password'}),
        help_text='Confirm your password',
    )
    user_type = forms.ChoiceField(
        choices=CustomUser.USER_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label="Account Type"
    )

    class Meta:
        model = CustomUser
        fields = ('username', 'password1', 'password2', 'user_type')
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Choose a username'}),
        }

    def clean_password2(self):
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords do not match.")
        return password2
