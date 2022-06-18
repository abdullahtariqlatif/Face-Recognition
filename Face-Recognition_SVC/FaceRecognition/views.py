# from django.contrib.auth.models import User
import threading
import time

from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect
from django.views.decorators import gzip
from . import models
import shutil
from datetime import datetime

# import libraries for traininig
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os
import imutils
# library to read file and shw data on web page
from prettytable import prettytable
import pandas as pd


# from django.contrib.auth.decorators import permission_required


# Create your views here.

def home(request):
    return render(request, 'home.html')


def signup(request):
    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        # superuser = User.objects.create_superuser(username=username, email=email, password=password)
        # superuser.save()
        data = models.admindata(username=username, email=email, password=password)
        data.save()
        # print("user created")
        return redirect('/')
    return render(request, 'signup.html')


# @permission_required('auth.view_user')
def login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        users = models.admindata.objects.all()
        for user in users:
            if user.username == username and user.password == password:
                return render(request, 'navbar.html')
    return render(request, 'login.html')


# #to capture video class
# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         (self.grabbed, self.frame) = self.video.read()
#         threading.Thread(target=self.update, args=()).start()
#
#     def __del__(self):
#         self.video.release()
#
#     def get_frame(self):
#         image = self.frame
#         _, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()
#
#     def update(self):
#         while True:
#             (self.grabbed, self.frame) = self.video.read()
#
# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# @gzip.gzip_page
def navbar(request):
    # try:
    #     cam = VideoCamera()
    #     return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    # except:
    #     pass
    return render(request, 'navbar.html')


def addstudent(request):
    if request.method == "POST":
        name = request.POST['name']
        fathername = request.POST['fathername']
        address = request.POST['address']
        gender = request.POST['gender']
        department = request.POST['department']
        birthdate = request.POST['birthdate']
        id = request.POST['id']
        email = request.POST['email']
        studentdata = models.student(name=name, fathername=fathername, address=address, gender=gender,
                                     department=department, birthdate=birthdate, id=id, email=email)
        studentdata.save()
        return redirect('record')
    data = models.department.objects.all()
    return render(request, 'addstudent.html', {'record': data})


def record(request):
    data = models.student.objects.all()
    return render(request, 'record.html', {"record": data})


def delete(request, id, name):
    data = models.student.objects.get(id=id, name=name)
    data.delete()
    path = "F:\\4th semester\\AOA(SIR KASHIF)\\Project\\Images\\" + name
    if os.path.isdir(path):
        shutil.rmtree(path)
    return redirect('record')


def update(request, id):
    print(id)
    if request.method == "POST":
        name = request.POST['name']
        fathername = request.POST['fathername']
        address = request.POST['address']
        gender = request.POST['gender']
        department = request.POST['department']
        # section = request.POST['section']
        birthdate = request.POST['birthdate']
        id = request.POST['id']
        email = request.POST['email']
        studentdata = models.student(name=name, fathername=fathername, address=address, gender=gender,
                                     department=department, birthdate=birthdate, id=id, email=email)
        studentdata.save()
        return redirect('record')
    data = models.student.objects.get(id=id)
    return render(request, 'update.html', {"record": data})


def captureimage(request, id, name):
    imgcounter = 1
    models.student.objects.get(id=id, name=name)
    cur_path = os.getcwd()
    imgfolder_path = os.path.join(cur_path,'Images')
    os.chdir(imgfolder_path)
    SubFolder = name
    path3 = imgfolder_path + "\\" + SubFolder
    if os.path.isdir(path3):
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("Face Recognition", frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                print("Escape entered, closing the app")
                break
            elif k % 256 == 32:
                img_name = name + "_{}.png".format(imgcounter)
                cv2.imwrite(os.path.join(path3, img_name), frame)
                imgcounter += 1

        cv2.waitKey(1)
        cam.release()
        cv2.destroyAllWindows()
    else:
        os.mkdir(SubFolder)
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("Face Recognition", frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                print("Escape entered, closing the app")
                break
            elif k % 256 == 32:
                img_name = name + "_{}.png".format(imgcounter)
                cv2.imwrite(os.path.join(path3, img_name), frame)
                imgcounter += 1
        cv2.waitKey(1)
        cam.release()
        cv2.destroyAllWindows()
    return redirect('record')


def train(request):
    print("==================================================================")
    curr_path = os.getcwd()  # Current path of project directory

    # Importing Face Detection Model
    print("Loading Face Detection Model Files")
    proto_path = os.path.join(curr_path, 'Model Files', 'FD_deploy.prototxt')
    caffe_model_path = os.path.join(curr_path, 'Model Files', 'FD_res10_300x300_ssd_iter_140000.caffemodel')
    print("==================================================================")

    # Creating a Face Detector
    print("Creating a Face Detector")
    face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=caffe_model_path)
    print("==================================================================")

    # Importing Face Recognition Model
    print("Loading Face Recognition Model")
    recognition_model = os.path.join(curr_path, 'Model Files', 'FR_openface_nn4.small2.v1.t7')
    print("==================================================================")

    # Creating a Face Recognizer
    print("Creating a Face Recognizer")
    face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)
    print("==================================================================")

    # Getting Root Location For Images
    print("Getting Root Location For Images")
    data_base_path = os.path.join(curr_path, 'Images')
    print("==================================================================")

    # Getting all the paths of images in the Database Folder
    print("Getting all the paths of images in the Database Folder")
    filenames = []
    for path, subdirs, files in os.walk(data_base_path):
        for name in files:
            filenames.append(os.path.join(path, name))
    print("==================================================================")

    print("File names are:")
    print("--------------------------------------------------------------")
    print(filenames)
    print("==================================================================")

    # Reading Images
    print(
        "Reading Images using cv2\nResize Images\nSetting Image Blob\nPassing Image Blog to Face Detection Model \nSaving all faces in a list")
    face_embeddings = []
    face_names = []

    for (i, filename) in enumerate(filenames):
        print("Processing image {}".format(filename))

        image = cv2.imread(filename)
        image = imutils.resize(image, width=600)

        # Getting height and width of image
        (h, w) = image.shape[:2]

        # Setting the image blob
        image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False,
                                           False)

        # Passing the image_blob in face detector
        face_detector.setInput(image_blob)
        # Fetch the results # This has the Results of our face detector model
        face_detections = face_detector.forward()

        # Checking confidence ,If the detected image is really a face or not
        i = np.argmax(face_detections[0, 0, :, 2])
        confidence = face_detections[0, 0, i, 2]

        if confidence >= 0.5:
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # This contains only the face
            face = image[startY:endY, startX:endX]

            # Creating a face blob to pass to face recognizer model
            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0), True, False)

            # Passing face_blob to face recognizer model
            face_recognizer.setInput(face_blob)
            # Fetch the results # This has the Results of our face recognizer model
            face_recognitions = face_recognizer.forward()

            # Getting name of images according to folders
            name = filename.split(os.path.sep)[-2]

            # Saving the image data and names in lists
            face_embeddings.append(face_recognitions.flatten())
            face_names.append(name)

    # Saving the lists in Dictionary
    data = {"embeddings": face_embeddings, "names": face_names}

    print("Encoding Labels")
    le = LabelEncoder()
    labels = le.fit_transform((data["names"]))
    print("==================================================================")

    print("Training Model")
    # Training by fitting embeddings in model
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)
    print("==================================================================")

    print("Saving Recognizer model results in pickle file")
    f = open('Recognizer.pickle', "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
    print("==================================================================")

    print("Saving image labels in pickle file")
    f = open("Labels.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()
    print("==================================================================")
    return redirect('navbar')


def adddepartment(request):
    if request.method == "POST":
        dept = request.POST['dept']
        department = models.department(dept=dept)
        department.save()
        return redirect('departments')
    return render(request, 'adddepartment.html')


# def addsection(request, dept):
#     if request.method == "POST":
#         dept = request.POST['dept']
#         section = request.POST['section']
#         sections = models.sec(dept=dept, section=section)
#         sections.save()
#         return redirect('departments')
#     data = models.department.objects.get(dept=dept)
#     return render(request, 'addsection.html', {'depts': data})


def departments(request):
    data = models.department.objects.all()
    return render(request, 'departments.html', {"depts": data})


def deletedept(request, dept):
    data = models.department.objects.get(dept=dept)
    data.delete()
    return redirect('departments')


def testing(request):
    data = models.student.objects.all()
    print("==================================================================")
    curr_path = os.getcwd()  # Current path of project directory
    camera_port = 0  # Port for camera

    # Importing Face Detection Model
    print("Loading Face Detection Model Files")
    proto_path = os.path.join(curr_path, 'Model Files', 'FD_deploy.prototxt')
    caffe_model_path = os.path.join(curr_path, 'Model Files', 'FD_res10_300x300_ssd_iter_140000.caffemodel')
    print("==================================================================")

    # Creating a Face Detector
    print("Creating a Face Detector")
    face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=caffe_model_path)
    print("==================================================================")

    # Importing Face Recognition Model
    print("Loading Face Recognition Model")
    recognition_model = os.path.join(curr_path, 'Model Files', 'FR_openface_nn4.small2.v1.t7')
    print("==================================================================")

    # Creating a Face Recognizer
    print("Creating a Face Recognizer")
    face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)
    print("==================================================================")

    # Reading Pickle files of Recognizer and Lables
    print("Reading Pickle files of Recognizer and Lables")
    recognizer = pickle.loads(open('Recognizer.pickle', "rb").read())
    le = pickle.loads(open('Labels.pickle', "rb").read())
    print("==================================================================")

    print("Starting test video file")
    # vc = cv2.VideoCapture(camera_port) # Gives some Warning with this !!!!!
    vc = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    # Check if the webcam is opened correctly
    if not vc.isOpened():
        raise IOError("Cannot open webcam")

    time.sleep(1)

    print("Starting Detection and Recognition Loop")
    while True:

        # Reading Each frame
        ret, frame = vc.read()

        # Resizing frames
        frame = imutils.resize(frame, width=600)

        # Getting height and width of image
        (h, w) = frame.shape[:2]

        # Setting the image blob
        # image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

        try:
            image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                               (104.0, 177.0, 123.0), False, False)
        except cv2.error as e:
            print('Invalid frame, Can not Detect Face!')
        # cv2.waitKey()

        # Passing the image_blob in face detector
        face_detector.setInput(image_blob)
        # Fetch the results # This has the Results of our face detector model
        face_detections = face_detector.forward()

        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]

            if confidence >= 0.5:
                # Setting face box dimensions
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Getting boundaries of face
                face = frame[startY:endY, startX:endX]

                # Face dimensions
                (fH, fW) = face.shape[:2]

                # Creating a face blob to pass to face recognizer model
                # face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0), True, False)

                try:
                    face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0), True, False)
                except cv2.error as e:
                    print('Invalid frame, Cannot Recognize Face!!')
                # cv2.waitKey()

                # Passing face_blob to face recognizer model
                face_recognizer.setInput(face_blob)

                # Fetch the results # This has the Results of our face recognizer model
                vec = face_recognizer.forward()

                # Predicting
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)

                # Probability
                proba = preds[j]
                name = le.classes_[j]
                for dataa in data:
                    if name == dataa.name:
                        with open('Attendance.csv', 'r+') as f:
                            myDataList = f.readlines()
                            namesList = []
                            for line in myDataList:
                                entry = line.split(',')
                                namesList.append(entry[0])
                            if name not in namesList:
                                now = datetime.now()
                                dtstring = now.strftime('%d-%m-%Y %H:%M:%S')
                                f.writelines(f'\n{dataa.name},{dtstring},{dataa.id},{dataa.department},"Present"')
                #         att = models.attendance(name=dataa.name, id=dataa.id, attendance="Present", time=datetime)
                #         att.save()
                text = "{}: {:.2f}".format(name, proba * 100)

                # Putting box and name around face
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # Showing the frames
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Press q to quit
        if key == ord('q'):
            print("Program Ended by pressing q")
            break

    vc.release()  # Error is here
    # Destroy all windows when q is pressed
    cv2.destroyAllWindows()
    return render(request, 'navbar.html')


def viewattendance(request, id):
    a = pd.read_csv("Attendance.csv")
    a.to_html("viewattendance.html")
    # print(a.to_html())
    # print(a.at[0])
    # file = open("Attendance.csv", 'r')
    # read = file.readlines()
    # l1 = read[0]
    # l1 = l1.split(',')
    # t = prettytable([l1[0], l1[1], l1[2], l1[3], l1[4]])
    # for i in range(1, len(read)):
    #     t.add_row(len[i].split(','))
    # code = t.get_html_string()
    # html_file = open("viewattendance", 'w')
    # html_file = html_file.write(code)
    return render(request, 'viewattendance.html')
