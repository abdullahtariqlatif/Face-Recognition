# DigitalSreeni

from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns

###########################################################################################################

print("=================================================================================================")
print("\n")

print(os.listdir("D:\PUCIT\Attendence & Security System\Face_Recognition_Module\Training_Images"))
print("\n")

SIZE = 128

print("Training")
print("\n")

train_images = []
train_labels = []
for directory_path in glob.glob(
        "D:\PUCIT\Attendence & Security System\Face_Recognition_Module\Training_Images\Training\\*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)
    print("\n")

train_images = np.array(train_images)
train_labels = np.array(train_labels)

print("\ntrain_images data : ===========================================\n", len(train_images))
print("\ntrain_labels data : ===========================================\n", len(train_labels))
print("\n")

print("\ntrain_images data : ===========================================\n", train_images)
print("\ntrain_labels data : ===========================================\n", train_labels)
print("\n")

print("Training Ended")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Testing")

# test
test_images = []
test_labels = []
for directory_path in glob.glob(
        "D:\PUCIT\Attendence & Security System\Face_Recognition_Module\Training_Images\Validation\\*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

print("\ntest_images data : ===========================================\n", len(test_images))
print("\ntest_labels data : ===========================================\n", len(test_labels))
print("\n")

print("\ntest_images data : ===========================================\n", test_images)
print("\ntest_labels data : ===========================================\n", test_labels)
print("\n")

print("Testing Ended")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Encoding from text to integers using sklearn preprocessing")

# Encode labels from text to integers.
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

print("sklearn preprocessing Ended")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Split data into test and train datasets (already split but assigning to meaningful convention")

# Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

print("Splitting Ended")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Normalizing pixel values to between 0 and 1")

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Normalizing Ended")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("One Hot Encoding Y Values for Neural Network")

# One hot encode y values for neural network.
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

print("One Hot Encoding Y Ended")
###########################################################################################################

print("\n")
print("Getting VGG16 Model with weights = imagenet")
from keras.applications.vgg16 import VGG16

# Load model without classifier/fully connected layers
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

print("Model Loaded")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Setting model layers to non trainable")

# Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
    layer.trainable = False

print("Layers are now non trainable")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Loading model summary")

VGG_model.summary()  # Trainable parameters will be 0

print("Summary Loaded")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Using Features from convolutional network for RF")

# Now, let us use features from convolutional network for RF
feature_extractor = VGG_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

print("Features Extracted")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Setting features to X_for_RF to use in model")

X_for_RF = features  # This is our X input to RF

print("Features Set")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Loading Random Forest Model")

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators=50, random_state=42)

print("Model loaded")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Fitting model with X_for_RF and y_train")

# Train the model on training data
RF_model.fit(X_for_RF, y_train)  # For sklearn no one hot encoding

print("Features Set")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Sending test data through feature extractor process")

# Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

print("Sending Done")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Predicting")

# Now predict using the trained RF model.
prediction_RF = RF_model.predict(X_test_features)

print("Predicting Done")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Inverse le transforming")

# Inverse le transform to get original label back.
prediction_RF = le.inverse_transform(prediction_RF)

print("Inverse le transformed")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Printing Accuracy")

# Print overall accuracy
from sklearn import metrics

print("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))
print("Accuracy = ", (metrics.accuracy_score(test_labels, prediction_RF)) * 100, "%")

print("Accuracy Printed")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Confusion Matrix to Verify Accuracy")

# Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_RF)
# print(cm)
sns.heatmap(cm, annot=True)

print("Accuracy Verified")
print("=================================================================================================")
###########################################################################################################

print("\n")
print("Check results on Randomly selected images")

# Check results on a few select images
n = np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)
input_img_feature = VGG_model.predict(input_img)
input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_RF = RF_model.predict(input_img_features)[0]
prediction_RF = le.inverse_transform([prediction_RF])  # Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])

print("Own Checking Ended")
print("=================================================================================================")
###########################################################################################################
