from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser, DatEntry

class CustomUserCreationForm(UserCreationForm):
    user_type = forms.ChoiceField(choices=CustomUser.USER_TYPES, required=True)

    class Meta:
        model = CustomUser
        fields = ('username', 'user_type')


class DatEntryForm(forms.ModelForm):
    class Meta:
        model = DatEntry
        fields = '__all__'