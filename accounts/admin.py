from django.contrib import admin
from .models import CustomUser

class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('username', 'user_type')
    fields = ('username', 'user_type')

# Register your models here.
admin.site.register(CustomUser, CustomUserAdmin)