import numpy as np
import pickle
import os
import cv2
import time
import imutils

print("==================================================================")
curr_path = os.getcwd() # Current path of project directory
camera_port = 0 # Port for camera

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
        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
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

            #Getting boundaries of face
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

vc.release() # Error is here
# Destroy all windows when q is pressed
cv2.destroyAllWindows()
