import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os
import imutils

print("==================================================================")
curr_path = os.getcwd() # Current path of project directory

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
data_base_path = os.path.join(curr_path, 'Database')
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
print("Reading Images using cv2\nResize Images\nSetting Image Blob\nPassing Image Blog to Face Detection Model \nSaving all faces in a list")
face_embeddings = []
face_names = []

for (i, filename) in enumerate(filenames):
    print("Processing image {}".format(filename))

    image = cv2.imread(filename)
    image = imutils.resize(image, width=600)

    # Getting height and width of image
    (h, w) = image.shape[:2]

    # Setting the image blob
    image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

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
        face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0), True, False)

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
