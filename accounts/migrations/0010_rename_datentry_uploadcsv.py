# Generated by Django 5.1.1 on 2024-09-23 02:27

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0009_alter_datentry_gender'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='DatEntry',
            new_name='UploadCSV',
        ),
    ]
