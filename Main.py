import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.models import  Model
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.models import load_model
import datetime
import pickle

pd.set_option('display.max_columns', None)
# Load the training and testing data
train_data = pd.read_csv("Train_data.csv")
test_data = pd.read_csv("Test_data.csv")

# Data preprocessing
train_data.fillna({
    'Gender': train_data['Gender'].mode()[0],
    'Married': train_data['Married'].mode()[0],
    'Dependents': train_data['Dependents'].mode()[0],
    'Self_Employed': train_data['Self_Employed'].mode()[0],
    'Credit_History': train_data['Credit_History'].mode()[0]
}, inplace=True)

test_data.fillna({
    'Gender': test_data['Gender'].mode()[0],
    'Married': test_data['Married'].mode()[0],
    'Dependents': test_data['Dependents'].mode()[0],
    'Self_Employed': test_data['Self_Employed'].mode()[0],
    'Credit_History': test_data['Credit_History'].mode()[0]
}, inplace=True)

train_data.fillna({
    'LoanAmount': train_data['LoanAmount'].median(),
    'Loan_Amount_Term': train_data['Loan_Amount_Term'].median(),
}, inplace=True)

test_data.fillna({
    'LoanAmount': test_data['LoanAmount'].median(),
    'Loan_Amount_Term': test_data['Loan_Amount_Term'].median(),
}, inplace=True)

# Encode categorical variables
cols_to_encode = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
le = LabelEncoder()
print(test_data.columns)
for col in cols_to_encode:
    print(type(train_data[col][0]))
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    with open(f'label_encoder_${col}.pkl', 'wb') as file:
        pickle.dump(le, file)


X = train_data.drop(columns=['Loan_ID', 'Loan_Status'])
y = train_data['Loan_Status']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
y_train_scaled = le.fit_transform(y_train)
y_val_scaled = le.transform(y_val)

print(X)
# Build the Keras model
model=Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)), ## HL1 Connected wwith input layer
    Dense(32,activation='relu'), ## HL2
    Dense(1,activation='sigmoid')  ## output layer
]

)
print(y_train,y_val)
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

print(X_train_scaled.shape,y_train.shape,X_val_scaled.shape,y_val.shape)
# Train the model
history=model.fit(
    X_train_scaled,y_train_scaled,validation_data=(X_val_scaled,y_val_scaled),epochs=100)
# Save the model to an H5 file
with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)


# Evaluate the model
loss, accuracy = model.evaluate(X_val_scaled, y_val_scaled)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# You can also use your other classifiers here and compare performance if needed.
# For example:
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)
logistic_preds = logistic_model.predict(X_val_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_val, logistic_preds))

model.save('loan_eligibility_model.h5')
print("Model saved as loan_eligibility_model.h5")


# You can also save your models here using other formats if needed.
