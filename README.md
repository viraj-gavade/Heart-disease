# Heart Disease Prediction Web Application

A machine learning-based web application that predicts the risk of heart disease based on various health parameters.

## Live Demo

**[Access the Live Application Here](https://heart-disease-2gln.onrender.com/)**

## Features

- Interactive web interface for entering health parameters
- Real-time prediction of heart disease risk with confidence levels
- Visual risk assessment with intuitive color-coded indicators
- Detailed dashboard with data visualizations to understand risk factors
- Educational information about heart disease causes and prevention
- Mobile-responsive design works on all devices

## Tech Stack

- **Backend**: FastAPI, Python
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Logistic Regression)
- **Visualization**: Matplotlib, Seaborn, Chart.js

## Installation

1. Clone this repository
   ```
   git clone https://github.com/yourusername/heart-disease.git
   cd heart-disease
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   uvicorn application:app --reload
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## How It Works

1. **Input Collection**: Users provide their health parameters through an intuitive form
2. **Data Processing**: The application processes and scales the input data
3. **Prediction**: A trained machine learning model predicts the likelihood of heart disease
4. **Visualization**: Results are displayed with intuitive visual representations
5. **Risk Assessment**: A detailed risk assessment is provided based on the prediction

## Deployment

This application is successfully deployed on Render: [https://heart-disease-2gln.onrender.com/](https://heart-disease-2gln.onrender.com/)

The application can also be deployed to other platforms:
- **Render**: Using the included `render.yaml` configuration
- **Railway**: Using the included `Procfile`
- **Docker**: Using the included `Dockerfile`

## Model Details

The heart disease prediction model is trained using a dataset of patient health metrics. The model achieves high accuracy in identifying potential heart disease cases.

Key features used in prediction:
- Age and gender
- Chest pain type
- Resting blood pressure
- Serum cholesterol levels
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate
- Exercise-induced angina
- ST depression induced by exercise
- Other cardiovascular indicators

## Directory Structure

- `application.py`: Main FastAPI application
- `Models/`: Contains the trained machine learning model
- `templates/`: HTML templates for the web interface
- `static/`: Static assets (CSS, JavaScript, images)
- `notebooks/`: Jupyter notebooks used for model development and analysis

## Future Improvements

- User accounts for tracking health metrics over time
- API integration for healthcare providers
- Enhanced data visualizations
- Integration with wearable device data

## License

© 2025 Viraj Gavade. All Rights Reserved.

## Connect with Me

- **GitHub**: [github.com/yourusername](https://github.com/viraj-gavade)
- **LinkedIn**: [linkedin.com/in/viraj-gavade](https://linkedin.com/in/viraj-gavade-dev)
- **Twitter**: [@viraj_gavade](https://twitter.com/viraj_gavade)
- **Portfolio**: [virajgavade.com](https://portfolio-viraj-gavades-projects.vercel.app/)

---

*This project was developed as part of ML Learning Projects. If you have any questions or suggestions, feel free to reach out!*
