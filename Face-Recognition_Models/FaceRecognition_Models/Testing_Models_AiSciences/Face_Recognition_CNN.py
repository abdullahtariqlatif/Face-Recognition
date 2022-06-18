from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split  # For training the data
from sklearn.svm import SVC  # Trained model
from sklearn.decomposition import PCA  # For creating a pipeline
from sklearn.pipeline import make_pipeline  # For creating a pipeline
from sklearn.ensemble import RandomForestClassifier  # Random forest it is I guess

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# --------------------------------------------------------------------------------------------------------------
# Getting datasets for faces
print("Getting face values form dataset")
faces = fetch_lfw_people(min_faces_per_person=101)
print("\nNumber of face classes ", len(faces.target_names))
print("\n")
# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------
# Showing related data for the dataset
print("\nNumber of face classes ", len(faces.target_names))
print("\n")
print("Face Target\n=============================================================\n", faces.target)
print("\n")
print("Face Target Names\n=============================================================\n", faces.target_names)
print("\n")
print("\nFace Data\n=============================================================\n", faces.data)
print("\n")
print("Unique Classes\n=============================================================\n ", np.unique(faces.target))
print("\n")
print("Face images\n=============================================================\n ", faces.images)
print("\n")
print("Face Description\n=============================================================\n ", faces.DESCR)
print("\n")
# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------
# Showing the images
i = 0
while i < len(faces):
    print(faces.target_names[i], "----->", plt.imshow(faces.images[i]), "----->", faces.images[i].shape)
    i += 1
# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------

print("\n")

X = faces.data
print("Data Points Matrix : ", X.shape)
print("\n")

Y = faces.target
print("Target faces : ", Y.shape)
print("\n")

print("Label no : ", Y[100])
print("Name of person : ", faces.target_names[Y[100]])
print("\n")
# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------
# Training data

# x_train = Data we should work on
# y_train = The corresponding labels of the x_train data
# x_test = Validation of data
# y_test = The corresponding labels of the x_test , Reserved for model evaluation
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)



print("\nThe x_train : \n" , x_train , "\nThe y_train : \n" , y_train)
print("\nThe x_test : \n" , x_test , "\nThe y_test : \n" , y_test)


# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------
# Building own classifier, Nearest neighbour classifier

# Getting a random test point from the data in the range of the y_test size
random_idx = np.random.randint(y_test.size)
# Getting a data point from the validation data set
x_t = x_test[random_idx]
# Now we need to know the label of this points
# To do that we find the nearest neighbour to it and it's label and so on
# x_t will be subtracted from each training sample in x_train
# If x_t is closest to some image than the difference would be small
# Take square to nullify the impact of sign!!!!
# Then we add to see how similar they are
# Adding all column wise , So , the axis is 1
# Now form all the values in the column we get the smallest one by using np.argmin()
nn_idx = np.argmin(((x_train - x_t) ** 2).sum(axis=1))
# After all the above the nn_idx will contain the closest neighbour to x_t
# Comparing found value with already know value
y_pred = y_train[nn_idx]  # Predicted value
y_true = y_test[random_idx]  # Actual value

# Printing the result
print("True: ", y_true, " Predictied: ", y_pred)
if y_true == y_pred:
    print("Success!!!!!!!!")
else:
    print("Missed!!!")
# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------
# Support Vector Machine Model
print("\nUsing Vector Machine Model now!!")

model = SVC(gamma='auto')
model.fit(x_train, y_train)  # Giving dataset
# Predicting using model function
y_pred = model.predict(x_test)
# Accuracy percentage calculation, How many times you guess right
acc = ((y_pred == y_test).sum() / y_test.size) * 100
print("Accuracy is : ", acc, "%")

# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------
# Creating Pipeline
print("\nUsing Vector Machine after creating Pipeline")

model_PCA = PCA(n_components=300)  # Reduce dimensions to 300
model_SVM = SVC(gamma='auto')  # Support vector machine
model_pipeline = make_pipeline(model_PCA, model_SVM)
model_pipeline.fit(x_train, y_train)  # Giving dataset
# Predicting using model function
y_pred = model_pipeline.predict(x_test)
# Accuracy percentage calculation, How many times you guess right
acc = ((y_pred == y_test).sum() / y_test.size) * 100
print("Accuracy is : ", acc, "%")
# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------
# Using Random Forest now

print("\nNow using Random Forest Classifier")

model_random = RandomForestClassifier(n_estimators=30)  # individual tress given as parameter
model_random.fit(x_train, y_train)  # Giving dataset
# Predicting using model function
y_pred = model_random.predict(x_test)
# Accuracy percentage calculation, How many times you guess right
acc = ((y_pred == y_test).sum() / y_test.size) * 100
print("Accuracy is : ", acc, "%")
