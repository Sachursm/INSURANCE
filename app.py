import numpy as np
from flask import Flask, request, render_template
from model import load_model

app = Flask(__name__)

# Load the model
model = load_model()

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_form():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])
        
        # Create input array
        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return render_template('output.html', prediction=prediction)
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)