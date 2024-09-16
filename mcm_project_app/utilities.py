from django.conf import settings
import os
def csvPathFileName():
    return os.path.join(settings.BASE_DIR, 'dataset/datasets.csv')