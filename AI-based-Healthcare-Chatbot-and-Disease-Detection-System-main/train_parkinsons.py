import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Load the data
parkinsons_data = pd.read_csv('parkinsons.csv')

# Separate features and target
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Save the model
pickle.dump(model, open('parkinsons_model.sav', 'wb'))

print("Model trained and saved successfully!") 