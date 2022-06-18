import matplotlib.pyplot as plt
import face_recognition
import numpy as np
import imutils
import time
import glob
import cv2
import os
from sklearn import metrics


from VGG16_Training import VGG_model, RF_model, le
from VGG16_Training import test_labels

###########################################################################################################


print("=================================================================================================")

root_dir = os.getcwd()
img_dir = os.path.join(root_dir, 'Training_Images')
train_img_dir = os.path.join(img_dir, 'Training', '*')
valid_img_dir = os.path.join(img_dir, 'Validation', '*')
print("\n")

SIZE = 128
camera_port = 0  # Port for camera
###########################################################################################################


###########################################################################################################
###########################################################################################################

# Importing Face Detection Model
print("Loading Face Detection Model Files")
proto_path = os.path.join(root_dir, 'Model Files', 'FD_deploy.prototxt')
caffe_model_path = os.path.join(root_dir, 'Model Files', 'FD_res10_300x300_ssd_iter_140000.caffemodel')
print("==================================================================")

# Creating a Face Detector
print("Creating a Face Detector")
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=caffe_model_path)
print("==================================================================")

###########################################################################################################

print("Starting Camera")
vc = cv2.VideoCapture(camera_port) # Gives some Warning with this !!!!!
# vc = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not vc.isOpened():
    raise IOError("Cannot open webcam")
time.sleep(1)

print("Starting Detection and Recognition Loop")
while True:

    # Reading Each frame
    ret, frame = vc.read()

    input_img_expand = np.expand_dims(frame, axis=0)

    print("*************************************************************************************************")
    print("Predicting in VGG Model")
    # Predicting image by passing in VGG model
    try:
        input_img_feature = VGG_model.predict(input_img_expand)
        print("*************************************************************************************************")
        print("Reshaping")
        # Reshaping # Needs explanation
        input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)

        print("*************************************************************************************************")
        print("Predicting input features in Random Forest model")
        # Predicting input features in Random Forest model
        prediction_RF_R = RF_model.predict(input_img_features)[0]

        print("*************************************************************************************************")
        print("Le inversing")
        prediction_RF_R_label = le.inverse_transform([prediction_RF_R])  # Reverse the label encoder to original name

        print("\nThe prediction for this image is: ", prediction_RF_R_label)

        print("*************************************************************************************************")
        print("Calculating Probability")
        # Probability
        probability = metrics.accuracy_score(test_labels, prediction_RF_R_label)
        name = prediction_RF_R_label

        print("*************************************************************************************************")
        print("Saving Name and Probability")
        # Displaying Probability
        text = "{}: {:.2f}".format(name.upper(), probability * 100)

        print("*************************************************************************************************")
        print("Setting Boundary data and rectangle")
        faceLoc = face_recognition.face_locations(frame)
        # face location has four values Top,Right,Bottom,Left
        y1, x2, y2, x1 = faceLoc
        # Adjusting the location because we resized it to 25%
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Drawing a rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
        cv2.putText(frame, text, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    except:
        print("Invalid Frame")















    # # --------------------------------------------------------------------------------------------------------------
    # # Shrinking original cameraImg to 0.25 times the size along x & y axis
    # cameraImgSmall = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    # # cameraImgSmall = cv2.cvtColor(cameraImgSmall, cv2.COLOR_BGR2RG)
    #
    # # frame = imutils.resize(frame, width=600)
    #
    # # Detecting faces in image in case of multiple persons in an image
    # faces_detected = face_recognition.face_locations(frame)


    # for face_img in faces_detected:
    #
    #     print("*************************************************************************************************")
    #     print("Predicting in VGG Model")
    #     # Predicting image by passing in VGG model
    #     input_img_feature = VGG_model.predict(face_img)
    #
    #     print("*************************************************************************************************")
    #     print("Reshaping")
    #     # Reshaping # Needs explanation
    #     input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
    #
    #     print("*************************************************************************************************")
    #     print("Predicting input features in Random Forest model")
    #     # Predicting input features in Random Forest model
    #     prediction_RF_R = RF_model.predict(input_img_features)[0]
    #
    #     print("*************************************************************************************************")
    #     print("Le inversing")
    #     prediction_RF_R_label = le.inverse_transform([prediction_RF_R])  # Reverse the label encoder to original name
    #
    #     print("\nThe prediction for this image is: ", prediction_RF_R_label)
    #
    #     print("*************************************************************************************************")
    #     print("Calculating Probability")
    #     # Probability
    #     probability = metrics.accuracy_score(test_labels, prediction_RF_R_label)
    #     name = prediction_RF_R_label
    #
    #     print("*************************************************************************************************")
    #     print("Saving Name and Probability")
    #     # Displaying Probability
    #     text = "{}: {:.2f}".format(name.upper(), probability * 100)
    #
    #     print("*************************************************************************************************")
    #     print("Setting Boundary data and rectangle")
    #     faceLoc = face_recognition.face_locations(face_img)
    #     # face location has four values Top,Right,Bottom,Left
    #     y1, x2, y2, x1 = faceLoc
    #     # Adjusting the location because we resized it to 25%
    #     y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    #
    #     # Drawing a rectangle around the face
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    #     cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
    #     cv2.putText(frame, text, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)













    print("=========================================================================================")
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

###########################################################################################################
