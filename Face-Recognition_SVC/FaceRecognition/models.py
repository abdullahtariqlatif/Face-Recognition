from django.db import models


# Create your models here.
class admindata(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField()
    password = models.CharField(max_length=10)


class student(models.Model):
    name = models.CharField(max_length=30)
    fathername = models.CharField(max_length=30)
    address = models.CharField(max_length=70)
    gender = models.CharField(max_length=20)
    department = models.CharField(max_length=50)
    section = models.CharField(max_length=10)
    birthdate = models.DateField()
    id = models.CharField(max_length=50, primary_key=True, unique=True)
    email = models.EmailField(unique=True)


# class sec(models.Model):
#     department = models.CharField(max_length=50)
#     sect = models.CharField(max_length=50)


class department(models.Model):
    dept = models.CharField(max_length=40, primary_key=True)


# class attendance(models.Model):
#     id = models.CharField(max_length=50, primary_key=True, unique=True)
#     name = models.CharField(max_length=30)
#     attendance = models.CharField(max_length=30, default="Absent")
#     time = models.TimeField(auto_now=True)
