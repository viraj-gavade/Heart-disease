import pickle
import numpy as np
from application import model, scaler

# Define several test cases with different expected outcomes
test_cases = [
    # Format: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    # Case 1: Very high risk case that should show Presence
    np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]),
    # Case 2: Another high risk case
    np.array([[70, 1, 3, 170, 250, 1, 0, 100, 1, 3.5, 1, 2, 2]]),
    # Case 3: Low risk case
    np.array([[35, 0, 0, 110, 160, 0, 0, 130, 0, 0.0, 0, 0, 0]]),
    # Case 4: Moderate risk case
    np.array([[55, 1, 1, 140, 220, 0, 0, 145, 1, 1.5, 1, 0, 1]]),
    # Case 5: High risk case with multiple risk factors
    np.array([[65, 1, 2, 160, 280, 1, 2, 108, 1, 2.0, 2, 3, 2]])
]

# Process each test case
for i, features in enumerate(test_cases):
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)
    
    # Get result
    result = "Presence" if prediction[0] == 1 else "Absence"
    
    # Calculate confidence according to our new logic
    if prediction[0] == 1:
        confidence = round(probability[0][1] * 100, 2)
    else:
        confidence = round(probability[0][0] * 100, 2)
    
    # Always calculate risk as probability of disease
    risk_prob = round(probability[0][1] * 100, 2)
    
    # Determine risk level based on risk_prob
    if risk_prob < 30:
        risk_level = "Low Risk"
    elif risk_prob < 70:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"
    
    print(f"Case {i+1}:")
    print(f"  Prediction: {result} of heart disease")
    print(f"  Confidence in prediction: {confidence}%")
    print(f"  Risk probability: {risk_prob}% ({risk_level})")
    print(f"  Raw probabilities: [No disease: {probability[0][0]:.4f}, Disease: {probability[0][1]:.4f}]")
    print()
