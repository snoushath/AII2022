from util import load_alz
import numpy as np

# read the data which is also normalized.
x_train, y_train, x_test, y_test = load_alz()

# make it 3D (for VGG16)
x_train = np.expand_dims(x_train, axis=-1)
x_train = np.repeat(x_train, 3, axis=3)

x_test = np.expand_dims(x_test, axis=-1)
x_test = np.repeat(x_test, 3, axis=3)

from tensorflow.keras.applications.vgg16 import VGG16
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(176, 176, 3))

# Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
    layer.trainable = False

# Now, let us use features from convolutional network for RF
print("extracting VGG16 features")
feature_extractor=VGG_model.predict(x_train)
print("reshaping features")
features = feature_extractor.reshape(feature_extractor.shape[0], -1)
print(feature_extractor.shape)
print(features.shape)

X_for_KNN = features #This is our X input to RF

print("model-1:KNN")
from sklearn.neighbors import KNeighborsClassifier
model1_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

# Train the model on training data
model1_KNN.fit(X_for_KNN, y_train) # For sklearn no one hot encoding

print("predicting")
#Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

print("predicting for test")
# Now predict using the trained RF model.
prediction1 = model1_KNN.predict(X_test_features)

# model-2

print("model-2:svm")
from sklearn.svm import LinearSVC
model2_SVM = LinearSVC(max_iter=100)

model2_SVM.fit(X_for_KNN, y_train) # For sklearn no one hot encoding
prediction2 = model2_SVM.predict(X_test_features)

#model-3
print("model-3:RF")
from sklearn.ensemble import RandomForestClassifier
model3_RF = RandomForestClassifier(n_estimators=50, random_state=42)
model3_RF.fit(X_for_KNN, y_train) # For sklearn no one hot encoding

#Now predict using the trained RF model.
prediction3 = model3_RF.predict(X_test_features)

#model-4
import xgboost as xgb
print("XGBoost")
model4_XGB = xgb.XGBClassifier()
model4_XGB.fit(X_for_KNN, y_train)  # For sklearn no one hot encoding
prediction4 = model4_XGB.predict(X_test_features)

from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test, prediction1)
accuracy2 = accuracy_score(y_test, prediction2)
accuracy3 = accuracy_score(y_test, prediction3)
accuracy4 = accuracy_score(y_test, prediction4)

print('Accuracy Score for model1 (KNN)= ', accuracy1)
print('Accuracy Score for model2 (SVM)= ', accuracy2)
print('Accuracy Score for model3 (RF) = ', accuracy3)
print('Accuracy Score for model3 (XGBoost) = ', accuracy4)
#Ensemble Prediction

print("ensemble accuracy prediction")
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[('knn', model1_KNN), ('svm', model2_SVM), ('RF',model3_RF), ('XGB', model4_XGB)], voting='hard')
model.fit(X_for_KNN,y_train)
ensemble_accuracy = model.score(X_test_features,y_test)
print("Ensemble accuracy is:",ensemble_accuracy)
