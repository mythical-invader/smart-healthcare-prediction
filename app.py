from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd 

app = Flask(__name__)
CORS(app) 

print("Loading model and scaler...")
try:
    model = joblib.load('diabetes_model.pkl')
    sc = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Error loading model files: {e}")
    print("Make sure 'diabetes_model.pkl' and 'scaler.pkl' are in this folder!")

LABELS = {0: 'Non-Diabetic', 1: 'Pre-Diabetic', 2: 'Diabetic'}

RECS = {
    0: 'No immediate risk. Maintain balanced diet and annual screening.',
    1: 'Pre-Diabetic risk. 150 min/week exercise + dietitian consult.',
    2: 'Diabetic risk confirmed. Immediate medical review required.',
}

TRAIN_MEDIANS = {
    'glucose': 117.0, 
    'bp': 72.0, 
    'skin': 29.0, 
    'insulin': 125.0, 
    'bmi': 32.3
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']   
        preg, glucose, bp, skin, insulin, bmi, pedigree, age = data

        glucose = TRAIN_MEDIANS['glucose'] if glucose == 0 else glucose
        bp = TRAIN_MEDIANS['bp'] if bp == 0 else bp
        skin = TRAIN_MEDIANS['skin'] if skin == 0 else skin
        insulin = TRAIN_MEDIANS['insulin'] if insulin == 0 else insulin
        bmi = TRAIN_MEDIANS['bmi'] if bmi == 0 else bmi

        glucose_bmi = glucose * bmi
        age_bmi = age * bmi
        insulin_glucose = insulin / (glucose + 1e-5)

        final_features = [preg, glucose, bp, skin, insulin, bmi, pedigree, age, 
                          glucose_bmi, age_bmi, insulin_glucose]

        feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 
                           'Glucose_BMI', 'Age_BMI', 'Insulin_Glucose']
        
        input_df = pd.DataFrame([final_features], columns=feature_columns)

        X_scaled = sc.transform(input_df)
        pred = int(model.predict(X_scaled)[0])

        prob = model.predict_proba(X_scaled)[0].tolist()

        return jsonify({
            'class_label': LABELS[pred],
            'probabilities': {LABELS[i]: f'{p*100:.1f}%' for i, p in enumerate(prob)},
            'recommendation': RECS[pred],
        })
        
    except Exception as e:

        print(f"\n--- PYTHON CRASHED ---\nError Details: {str(e)}\n----------------------\n") 
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the server on port 5000
    print("Starting Flask API for Smart Healthcare Prediction System...")
    app.run(debug=False, host='0.0.0.0', port=5000)