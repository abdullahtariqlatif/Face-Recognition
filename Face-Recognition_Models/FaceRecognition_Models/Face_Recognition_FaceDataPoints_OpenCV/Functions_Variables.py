# Header File Containing all variables and functions

# Libraries
import os
import cv2
import face_recognition

# --------------------------------------------------------------------------------------------------------------
# Folders to check for images

folders = [
    ## Add folder paths containing images for each person
]

# Storing the path of directory in which images/folders/files are stored
rootFolderPath = ""
# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
# Creating a list to store names of images extracted from a list of items in the scanned folders
imagesList = []

# Creating a list to store names of classes(group having same person images)
# This will be used inside a function
classNames = []
# This will be the global list
allClassNames = []

# Known Encoded images will be saved in this list
allKnownEncodedImages = []

# count to keep track of no.of items encoded
count = 0
# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
# Returns images with in a given folder having the extension .jpeg and .jpg
def loadImagesFromFolder(folderToSearch):
    imagesTemp = []
    classNames.clear()

    for filename in os.listdir(folderToSearch):

        if any([filename.endswith(x) for x in ['.jpeg', '.jpg']]):
            img = cv2.imread(os.path.join(folderToSearch, filename))

            if img is not None:
                imagesTemp.append(img)
                # Saving files/images them in 'classNames' without extension
                classNames.append(os.path.splitext(filename)[0])
                allClassNames.append(os.path.splitext(filename)[0])

    return imagesTemp


# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
# Creating a function to encode images
# Function for a separate thread
def encodeImages(imagesToEncode):
    encodedImagesTemp = []
    global count
    for img in imagesToEncode:
        # Coloring current img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # encoding current img
        encodeCurImg = face_recognition.face_encodings(img)[0]
        # storing encoded images
        encodedImagesTemp.append(encodeCurImg)
        print("Image no.", count + 1, "encoded and appended in list")
        count = count + 1
    return encodedImagesTemp
# --------------------------------------------------------------------------------------------------------------
