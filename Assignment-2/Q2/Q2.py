import os
import cv2
import numpy as np
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(script_dir))  # go up to root level

X_train = []
y_train = []
X_test = []
y_test = []

# getting the train data
train_path = os.path.join(base_dir, 'Assignment-2', 'Q2', 'train', '*', '*.jpg')
for i, address in enumerate(glob.glob(train_path)):
    img = cv2.imread(address)
    img = cv2.resize(img, (64, 64))
    img = img.flatten()
    img = img/255.0
    X_train.append(img)
    y_train.append(os.path.basename(os.path.dirname(address)))
    if i % 100 == 0:
        print(f'Processed {i} training images')


# getting the test data
test_path = os.path.join(base_dir, 'Assignment-2', 'Q2', 'test', '*', '*.jpg')
for i, address in enumerate(glob.glob(test_path)):
    img = cv2.imread(address)
    img = cv2.resize(img, (64, 64))
    img = img.flatten()
    img = img/255.0
    X_test.append(img)
    y_test.append(os.path.basename(os.path.dirname(address)))
    if i % 2 == 0:
        print(f'Processed {i} testing images')

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
    
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred)


model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred = model_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred)

# the accuracy would remain the same on multiple runs of this code
# as we have static train and test data and are not using train-test split for randomness
print(f'Logistic Regression Accuracy: {accuracy_lr}')
print(f'KNN Accuracy: {accuracy_knn}')



models_dir = os.path.join(base_dir, 'Assignment-2', 'Q2', 'models')
os.makedirs(models_dir, exist_ok=True)


joblib.dump(model_lr, os.path.join(models_dir, 'logistic_regression_model.pkl'))
joblib.dump(model_knn, os.path.join(models_dir, 'knn_model.pkl'))
joblib.dump(le, os.path.join(models_dir, 'label_encoder.pkl'))
