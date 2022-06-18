
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os

###########################################################################################################

print("=================================================================================================")

root_dir = os.path.split(os.getcwd())
img_dir = os.path.join(root_dir, 'Training_Images')
train_img_dir = os.path.join(root_dir, 'Training_Images', 'Training', '*')
valid_img_dir = os.path.join(root_dir, 'Training_Images', 'Validation', '*')
print("\n")

SIZE = 128

# Training
print("Loading Training Images\n====================")
train_images = []
train_labels = []

for directory_path in glob.glob(train_img_dir):
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

print("Train_images Length :\n", len(train_images))
print("Train_labels Length :\n", len(train_labels))

print("=================================================================================================")
###########################################################################################################

# Testing
print("\n")
print("Loading Testing Images\n====================")
test_images = []
test_labels = []

for directory_path in glob.glob(valid_img_dir):
    fruit_label = directory_path.split("\\")[-1]
    print(fruit_label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)
    print("\n")

test_images = np.array(test_images)
test_labels = np.array(test_labels)

print("Test_images Length :\n", len(test_images))
print("Test_labels Length :\n", len(test_labels))

print("=================================================================================================")
print("\n")
###########################################################################################################
print("=================================================================================================")
print("Encoding from text to integers using sklearn preprocessing")

# Encode labels from text to integers.
le = preprocessing.LabelEncoder()
le.fit(test_labels)

test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)

train_labels_encoded = le.transform(train_labels)

print("=================================================================================================")
###########################################################################################################

print("Split data into test and train datasets (already split but assigning to meaningful convention")

# Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

print("=================================================================================================")
###########################################################################################################

print("Normalizing pixel values to between 0 and 1")

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

print("=================================================================================================")
###########################################################################################################

print("One Hot Encoding Y Values for Neural Network")

# One hot encode y values for neural network.
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

print("One Hot Encoding Y Ended")
print("=================================================================================================")
###########################################################################################################

print("Getting VGG16 Model with weights = imagenet")
from keras.applications.vgg16 import VGG16

# Load model without classifier/fully connected layers
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

print("Model Loaded")
print("=================================================================================================")
###########################################################################################################

print("Setting model layers to non trainable")

# Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
    layer.trainable = False

print("=================================================================================================")
###########################################################################################################

print("Using Features from convolutional network for RF")

# Now, let us use features from convolutional network for RF
feature_extractor = VGG_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

print("Features Extracted")
print("=================================================================================================")
###########################################################################################################

print("Setting features to X_for_RF to use in model")

X_for_RF = features  # This is our X input to RF

print("=================================================================================================")
###########################################################################################################

print("Loading Random Forest Model")

# RANDOM FOREST

RF_model = RandomForestClassifier(n_estimators=50, random_state=42)

print("=================================================================================================")
###########################################################################################################

print("Fitting model with X_for_RF and y_train")

# Train the model on training data
RF_model.fit(X_for_RF, y_train)  # For sklearn no one hot encoding

print("=================================================================================================")
###########################################################################################################

print("Sending test data through feature extractor process")

# Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

print("=================================================================================================")
###########################################################################################################

print("Predicting test data")

# Now predict using the trained RF model.
prediction_RF = RF_model.predict(X_test_features)

print("=================================================================================================")
###########################################################################################################

print("Inverse le transforming")

# Inverse le transform to get original label back.
prediction_RF = le.inverse_transform(prediction_RF)

print("=================================================================================================")
###########################################################################################################

print("Printing Accuracy")
# Print overall accuracy
print("Accuracy = ", (metrics.accuracy_score(test_labels, prediction_RF)) * 100, "%")

###########################################################################################################

print("*************************************************************************************************")
print("Predicting an image")

# Getting a random image
n = np.random.randint(0, x_test.shape[0])
img = x_test[n]

# Expanding dimensions
input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)

# Predicting image by passing in VGG model
input_img_feature = VGG_model.predict(input_img)

# Reshaping # Needs explanation
input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)

# Predicting input  features in Random Forest model
prediction_RF = RF_model.predict(input_img_features)[0]
prediction_RF = le.inverse_transform([prediction_RF]) # Reverse the label encoder to original name

print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])

print("*************************************************************************************************")
###########################################################################################################
