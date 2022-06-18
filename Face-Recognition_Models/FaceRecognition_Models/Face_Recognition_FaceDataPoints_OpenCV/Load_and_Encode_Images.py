# File to load and encode images in a list and save it in a file

# Libraries
import pickle
from Functions_Variables import folders, loadImagesFromFolder, classNames, allKnownEncodedImages, encodeImages, \
    allClassNames

# --------------------------------------------------------------------------------------------------------------
# Define a function to read images from folders and encode it

# Cycle through each folder
for folder in folders:
    # Take images from a folder and save it in a list
    imagesList = loadImagesFromFolder(folder)

    print("Items in 'classNames' : \n", classNames)

    print("Encoding images in folder...")
    # Calling the function to encode the images taken from a folder
    encodedImagesInFolder = encodeImages(imagesList)

    # Saving the encoded images in a global list
    allKnownEncodedImages.extend(encodedImagesInFolder)

    # print(type(allKnownEncodedImages))

    print("Encoding Complete!!!\n")
    # Printing the length of encoded images list
    print("Length of encoded image list in the folder is :", len(encodedImagesInFolder))

# Printing the length of allEncodedImagesKnown
print("Length of list containing all encoded images", len(allKnownEncodedImages))

# Printing the allClassNames
print("List containing all Class Names : ", allClassNames)

# print("one encoded image", allKnownEncodedImages[0])
# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
# Writing encoded data into file using pickle
with open('Encoded_Data_pickle_format', 'wb') as fp:
    pickle.dump(allKnownEncodedImages, fp)

print("Pickle file written successfully")
# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
# Writing encoded data into a text file

# open file
with open('Encoded_Data_Text_format', 'w+') as f:
    # write elements of list
    for items in allKnownEncodedImages:
        f.write('%s\n' % items)

print("Text file written successfully")

# close the file
f.close()
# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
# Returning the AllClassNames List to other program
def classNames():
    return allClassNames
# --------------------------------------------------------------------------------------------------------------
