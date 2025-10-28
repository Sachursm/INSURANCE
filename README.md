# 🧮 Insurance Cost Predictor

A Machine Learning-powered **Flask web app** that predicts insurance costs based on user inputs such as age, gender, BMI, number of children, smoking status, and region.

---

## 🚀 Overview

This project combines **Machine Learning** and **Web Development** to provide accurate insurance cost predictions through a clean and interactive web interface.

Users input their details, and the trained regression model instantly estimates their insurance charges.

---

# Insurance Cost Predictor

Users input their details, and the trained regression model instantly estimates their insurance charges.

## ✨ Features

- 🧠 **ML Model**: Predicts insurance cost using trained regression
- 🌐 **Flask Web App**: Easy-to-use interface for user interaction
- 🎨 **Beautiful UI**: Built using HTML + CSS templates
- 💾 **Saved Model**: Uses `insurance_model.pkl` for fast predictions
- 📊 **Dataset**: Includes `insurance.csv` for model training

## 📁 Project Structure

```
Insurance-Cost-Predictor/
│
├── __pycache__/          # Python cache files
├── templates/            # HTML pages for the Flask app
│   ├── welcome.html      # Home page
│   ├── input.html        # Input form page
│   └── output.html       # Result display page
│
├── app.py                # Flask main application
├── model.py              # ML model training script
├── insurance.csv         # Dataset
├── insurance_model.pkl   # Saved ML model
├── requirements.txt      # Python dependencies
└── .gitignore            # Ignored files
```
## ⚙️ Installation Guide

Follow these steps to run the project on your local system 👇

### 1️⃣ Clone the repository

```bash
git clone https://github.com/insurance-cost-predictor.git
cd insurance-cost-predictor
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Flask app

```bash
python app.py
```

Then open your browser and go to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
---
🧠 Model Details

Dataset Used: insurance.csv

Algorithm: Linear Regression

Target Variable: charges (predicted insurance cost)

Input Features:
```
Age, Gender, BMI, Children, Smoker, Region
```
The model is trained in model.py and saved as insurance_model.pkl, which is loaded by app.py for real-time predictions.
---

🖼️ App Screenshots

🏠 Welcome Page
![image alt](https://github.com/Sachursm/INSURANCE/blob/master/welcome.png?raw=true)

🧾 Input Form Page
![image alt](https://github.com/Sachursm/INSURANCE/blob/master/input.png?raw=true)

📊 Prediction Output Page
![image alt]()
