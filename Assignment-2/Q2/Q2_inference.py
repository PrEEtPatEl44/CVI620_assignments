import numpy as np
import joblib
import cv2
import glob
import matplotlib.pyplot as plt


model_knn = joblib.load('Q2/models/knn_model.pkl')
model_lr = joblib.load('Q2/models/logistic_regression_model.pkl')
le = joblib.load('Q2/models/label_encoder.pkl')

for i, address in enumerate(glob.glob('Q2/internet images/*')):
    img = cv2.imread(address)
    img = cv2.resize(img, (64, 64))
    img = img.flatten()
    img = img / 255.0
    img = np.array([img])

    # Predict using Logistic Regression
    pred_lr = model_lr.predict(img)
    
    # Predict using KNN
    pred_knn = model_knn.predict(img)

    print(f'Image: {address}, LR: {le.inverse_transform([pred_lr[0]])[0]}, KNN: {le.inverse_transform([pred_knn[0]])[0]}')




