from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login', views.login, name='login'),
    path('signup', views.signup, name='signup'),
    path('navbar', views.navbar, name='navbar'),
    path('addstudent', views.addstudent, name='addstudent'),
    path('record', views.record, name='record'),
    path('delete/<str:id>/<str:name>', views.delete, name='delete'),
    path('update/<str:id>', views.update, name='update'),
    path('captureimage/<str:id>/<str:name>', views.captureimage, name='captureimage'),
    path('train', views.train, name='train'),
    path('testing', views.testing, name='testing'),
    path('adddepartment', views.adddepartment, name='adddepartment'),
    path('departments', views.departments, name='departments'),
    path('deletedept/<str:dept>', views.deletedept, name='deletedept'),
    path('viewattendance/<str:id>', views.viewattendance, name='viewattendance')
]
