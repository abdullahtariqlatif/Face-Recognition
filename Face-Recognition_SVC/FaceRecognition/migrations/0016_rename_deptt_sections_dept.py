# Generated by Django 3.2.1 on 2021-06-07 08:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('FaceRecognition', '0015_rename_dept_sections_deptt'),
    ]

    operations = [
        migrations.RenameField(
            model_name='sections',
            old_name='deptt',
            new_name='dept',
        ),
    ]
