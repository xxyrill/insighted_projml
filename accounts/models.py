from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    USER_TYPES = (
        ('VPAA', 'Vice President of Academic Affairs'),
        ('IDEALS', 'Ideals'),
        ('HR', 'HR'),
        ('DEANS', 'Deans'),
    )

    username = models.CharField(max_length=150, unique=True)
    password = models.CharField(max_length=150)
    user_type = models.CharField(max_length=20, choices=USER_TYPES)
    # user_type = models.CharField(max_length=10, choices=USER_TYPES, default='DEANS')
    

    def __str__(self):
        return self.username

