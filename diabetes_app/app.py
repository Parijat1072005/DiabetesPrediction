from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
try:
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model files not found. Please run train_model.py first.")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from form
            pregnancies = float(request.form['pregnancies'])
            glucose = float(request.form['glucose'])
            blood_pressure = float(request.form['blood_pressure'])
            skin_thickness = float(request.form['skin_thickness'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = float(request.form['age'])

            # Organize features into a numpy array (1 row, 8 columns)
            features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            
            # Scale the features using the loaded scaler
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)
            
            # Determine result text
            if prediction[0] == 1:
                result_text = "The model predicts: DIABETIC"
                result_color = "red"
            else:
                result_text = "The model predicts: NON-DIABETIC"
                result_color = "green"

            return render_template('index.html', prediction_text=result_text, color=result_color)

        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}", color="black")

if __name__ == "__main__":
    app.run(debug=True)