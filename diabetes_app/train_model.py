import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

# 1. Load Data
url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
df = pd.read_csv(url)

# 2. Preprocessing (Imputing missing values as per your notebook)
# Replacing 0s with mean of non-zero values
mean_insulin = df['Insulin'][df['Insulin'] != 0].mean()
df['Insulin'] = df['Insulin'].replace(0, mean_insulin)

mean_skin = df['SkinThickness'][df['SkinThickness'] != 0].mean()
df['SkinThickness'] = df['SkinThickness'].replace(0, mean_skin)

# 3. Define X and y
x = df.iloc[:, :8]
y = df.iloc[:, -1]

# 4. Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# 5. Scaling (CRITICAL STEP)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 6. Train Model
clf = LogisticRegression()
clf.fit(x_train_scaled, y_train)

# 7. Save Model AND Scaler
if not os.path.exists('model'):
    os.makedirs('model')

pickle.dump(clf, open('model/model.pkl', 'wb'))
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))

print("Success! Model and Scaler have been saved to the 'model' folder.")