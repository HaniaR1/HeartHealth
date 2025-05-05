from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        cholesterol = float(request.form['cholesterol'])
        max_hr = float(request.form['max_hr'])

        data = np.array([[age, bp, cholesterol, max_hr]])
        prediction = model.predict(data)[0]

        result = "Heart Disease Risk" if prediction == 1 else "Low Risk"
        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
