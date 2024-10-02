from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser

class CustomUserCreationForm(UserCreationForm):
    user_type = forms.ChoiceField(choices=CustomUser.USER_TYPES, required=True)

    class Meta:
        model = CustomUser
        fields = ('username', 'user_type')

    def clean_password(self):
        return self.cleaned_data['password']