# Generated by Django 3.2.1 on 2021-07-12 16:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('FaceRecognition', '0021_attendance'),
    ]

    operations = [
        migrations.AlterField(
            model_name='attendance',
            name='attendance',
            field=models.CharField(default='Absent', max_length=30),
        ),
    ]
