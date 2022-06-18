# The main project
# Libraries
from __future__ import print_function
from Functions_Variables import allClassNames, allKnownEncodedImages
import cv2
import pickle
import numpy as np
import face_recognition

# --------------------------------------------------------------------------------------------------------------
# Reading from the pickle file
from Load_and_Encode_Images import classNames

with open('Encoded_Data_pickle_format', 'rb') as fp:
    allKnownEncodedImages = pickle.load(fp)
print("\nEncoded data read successfully from Pickle file \n")

allClassNames = classNames()
# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
# Enabling webcam
camera = cv2.VideoCapture(0)
# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
# Capturing video from camera
while True:

    success, cameraImg = camera.read()
    # cameraImg = camera.read()

    # Shrinking original cameraImg to 0.25 times the size along x & y axis
    cameraImgSmall = cv2.resize(cameraImg, (0, 0), None, 0.25, 0.25)

    cameraImgSmall = cv2.cvtColor(cameraImgSmall, cv2.COLOR_BGR2RGB)

    # Detecting faces in image in case of multiple persons in an image
    faceLocations = face_recognition.face_locations(cameraImgSmall)  # Original!!!!!!!!

    # Encoding the detected faces
    cameraImgEncoded = face_recognition.face_encodings(cameraImgSmall, faceLocations)  # Original!!!!!!!!!!

    # --------------------------------------------------------------------------------------------------------------
    # Comparing faces in camera input to the encoded faces with there face location provided

    for faceLoc, encodedFace in zip(faceLocations, cameraImgEncoded):
        # Comparing
        compareResult = face_recognition.compare_faces(allKnownEncodedImages, encodedFace)

        # Finding the distance between two images
        faceDist = face_recognition.face_distance(allKnownEncodedImages, encodedFace)
        print("The Distance between faces : ", faceDist)

        # Printing the allClassNames
        print("List containing all Class Names : ", allClassNames)

        # Finding minimum value of distance(possible face match)
        bestMatch = np.argmin(faceDist)
        print("Best Match :", bestMatch)

        # Printing the match if it exits
        if compareResult[bestMatch]:
            name = allClassNames[bestMatch].upper()
            nameForAttendance = name.split('_')
            print(nameForAttendance[0])

            # face location has four values Top,Right,Bottom,Left
            y1, x2, y2, x1 = faceLoc
            # Adjusting the location because we resized it to 25%
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # --------------------------------------------------------------------------------------------------------------
            # Drawing a rectangle around the face
            cv2.rectangle(cameraImg, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.rectangle(cameraImg, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(cameraImg, nameForAttendance[0], (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    # Displaying the found match original image
    cv2.imshow("Best Match", cameraImg)
    cv2.waitKey(1)
    # --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------
