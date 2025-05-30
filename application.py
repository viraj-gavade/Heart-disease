import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from io import BytesIO
import base64
import json
import uvicorn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Create FastAPI instance
app = FastAPI(title="Heart Disease Prediction App")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Load the saved model
import os
model_path = os.path.join(os.path.dirname(__file__), 'Models', 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'Models', 'scalar.pkl')
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Feature names (based on your dataset)
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                 age: int = Form(...),
                 sex: int = Form(...),
                 cp: int = Form(...),
                 trestbps: int = Form(...),
                 chol: int = Form(...),
                 fbs: int = Form(...),
                 restecg: int = Form(...),
                 thalach: int = Form(...),
                 exang: int = Form(...),
                 oldpeak: float = Form(...),
                 slope: int = Form(...),
                 ca: int = Form(...),
                 thal: int = Form(...)):
    
    # Create input features array
    features = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg, 
        thalach, exang, oldpeak, slope, ca, thal
    ]])
      # Scale features
    features_scaled = scaler.transform(features)
    
    # Get probabilities
    probability = model.predict_proba(features_scaled)
      # Calculate risk probability (probability of heart disease)
    # In our reversed model, higher risk_prob means LOWER risk
    risk_prob = round(probability[0][1] * 100, 2)
    
    # Make prediction based on probability[0][1] (probability of heart disease)
    # In our reversed model, if probability[0][1] > probability[0][0], predict Absence
    if probability[0][1] > probability[0][0]:
        result = "Absence"
        prediction = [1]
    else:
        result = "Presence"
        prediction = [0]# For UI/confidence display, use the probability corresponding to the predicted class
    if prediction[0] == 1:
        # Class 1: Presence of heart disease, use probability[0][1]
        heart_disease_prob = round(probability[0][1] * 100, 2)
    else:
        # Class 0: Absence of heart disease, use probability[0][0]
        heart_disease_prob = round(probability[0][0] * 100, 2)
          # Flag inconsistent predictions - for reversed model logic: 
    # Since prediction logic is reversed, mismatch is when result is "Presence" but risk_prob is high (>70%)
    # (which would mean low risk in our reversed model)
    prediction_mismatch = False
    if result == "Presence" and risk_prob > 70:
        prediction_mismatch = True
      # Prepare user data for template and visualization
    user_data = {
        "Age": age,
        "Sex": "Male" if sex == 1 else "Female",
        "Chest Pain Type": cp,
        "Resting Blood Pressure": trestbps,
        "Cholesterol": chol,
        "Fasting Blood Sugar": "Above 120 mg/dl" if fbs == 1 else "Below 120 mg/dl",
        "Resting ECG": restecg,
        "Max Heart Rate": thalach,
        "Exercise Induced Angina": "Yes" if exang == 1 else "No",
        "ST Depression": oldpeak,
        "Slope of Peak Exercise ST": slope,
        "Number of Major Vessels": ca,
        "Thal": thal
    }
    
    # Generate personalized chart
    patient_chart = generate_patient_chart(user_data)
      # Return the prediction to the template
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request,
            "prediction": result,
            "probability": heart_disease_prob,
            "risk_prob":  risk_prob,  # Add the actual risk probability
            "user_data": user_data,
            "patient_chart": patient_chart
        }
    )

# Function to generate visualizations from the heart disease dataset
def generate_visualizations():
    # Load the dataset
    heart_data_path = os.path.join(os.path.dirname(__file__), 'notebooks', 'heart.csv')
    heart_data = pd.read_csv(heart_data_path)
    
    # List to store visualization data with base64 encoded images
    viz_data = []
    
    # 1. Age distribution by heart disease
    plt.figure(figsize=(10, 6))
    sns.histplot(data=heart_data, x='age', hue='target', kde=True, palette=['red', 'green'])
    plt.title('Age Distribution by Heart Disease Status')  
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend(['Disease Present', 'No Disease'])
    plt.tight_layout()
    
    # Save to BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    viz_data.append({
        "title": "Age Distribution",
        "description": "Distribution of age among patients with and without heart disease.",
        "image_data": f"data:image/png;base64,{img_str}"
    })
    
    # 2. Gender distribution
    plt.figure(figsize=(8, 6))
    gender_counts = heart_data['sex'].value_counts()
    labels = ['Male', 'Female']
    colors = ['#457b9d', '#e63946']
    plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Gender Distribution')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    viz_data.append({
        "title": "Gender Distribution",
        "description": "Distribution of gender in the dataset.",
        "image_data": f"data:image/png;base64,{img_str}"
    })
    
    # 3. Heart disease by gender
    plt.figure(figsize=(10, 6))
    gender_disease = pd.crosstab(heart_data['sex'], heart_data['target'])
    gender_disease.plot(kind='bar', stacked=True, color=['red', 'green'])
    plt.title('Heart Disease by Gender')
    plt.xlabel('Gender (0 = Female, 1 = Male)')
    plt.ylabel('Count')
    plt.legend(['Disease Present', 'No Disease'])
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    viz_data.append({
        "title": "Heart Disease by Gender",
        "description": "Comparison of heart disease presence between males and females.",
        "image_data": f"data:image/png;base64,{img_str}"
    })
    
    # 4. Chest pain type and heart disease
    plt.figure(figsize=(10, 6))
    cp_disease = pd.crosstab(heart_data['cp'], heart_data['target'])
    cp_disease.plot(kind='bar', stacked=True, color=['red', 'green'])
    plt.title('Heart Disease by Chest Pain Type')
    plt.xlabel('Chest Pain Type')
    plt.ylabel('Count')
    plt.legend(['Disease Present', 'No Disease'])
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    viz_data.append({
        "title": "Heart Disease by Chest Pain Type",
        "description": "Relationship between different types of chest pain and heart disease presence.",
        "image_data": f"data:image/png;base64,{img_str}"
    })
    
    # 5. Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation = heart_data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Heart Disease Features')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    viz_data.append({
        "title": "Correlation Heatmap",
        "description": "Correlation matrix showing relationships between different features.",
        "image_data": f"data:image/png;base64,{img_str}"
    })
    
    return viz_data

# Generate charts for live data (for individual predictions)
def generate_patient_chart(patient_data):
    # Create a base64 encoded image of risk factors
    plt.figure(figsize=(10, 6))
    
    # Comparing patient data with average values
    heart_data_path = os.path.join(os.path.dirname(__file__), 'notebooks', 'heart.csv')
    heart_data = pd.read_csv(heart_data_path)
    
    # Select key metrics to compare
    metrics = ['age', 'trestbps', 'chol', 'thalach']
    labels = ['Age', 'Blood Pressure', 'Cholesterol', 'Max Heart Rate']
    
    # Calculate averages for each metric from the dataset
    avg_values = [
        heart_data['age'].mean(),
        heart_data['trestbps'].mean(),
        heart_data['chol'].mean(),
        heart_data['thalach'].mean()
    ]
    
    # Get patient values
    patient_values = [
        patient_data['Age'],
        patient_data['Resting Blood Pressure'],
        patient_data['Cholesterol'],
        patient_data['Max Heart Rate']
    ]
    
    # Normalize values for better comparison (0-1 scale)
    max_values = [80, 200, 600, 220]
    
    avg_values_norm = [avg_values[i]/max_values[i] for i in range(len(metrics))]
    patient_values_norm = [patient_values[i]/max_values[i] for i in range(len(metrics))]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    # Close the plot
    avg_values_norm = np.concatenate((avg_values_norm, [avg_values_norm[0]]))
    patient_values_norm = np.concatenate((patient_values_norm, [patient_values_norm[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    labels.append(labels[0])
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, avg_values_norm, 'o-', linewidth=2, label='Population Average')
    ax.fill(angles, avg_values_norm, alpha=0.25)
    ax.plot(angles, patient_values_norm, 'o-', linewidth=2, label='Your Data')
    ax.fill(angles, patient_values_norm, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_title('Your Health Metrics Compared to Average', size=14)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save to BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    # Encode to base64 for embedding in HTML
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_str

# Information page
@app.get("/info", response_class=HTMLResponse)
async def info(request: Request):
    return templates.TemplateResponse("info.html", {"request": request})

# About page
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

# Dashboard page with visualizations
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    # Generate visualizations
    viz_data = generate_visualizations()
    
    return templates.TemplateResponse(
        "dashboard.html", 
        {
            "request": request,
            "visualizations": viz_data
        }
    )

# API endpoint to get data for JavaScript visualizations
@app.get("/api/heart-data")
async def get_heart_data():
    heart_data_path = os.path.join(os.path.dirname(__file__), 'notebooks', 'heart.csv')
    heart_data = pd.read_csv(heart_data_path)
    data_json = heart_data.to_json(orient='records')
    return json.loads(data_json)

if __name__ == "__main__":
    uvicorn.run("application:app", host="0.0.0.0", port=8000, reload=True)