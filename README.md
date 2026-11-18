# ⚡ Smart Energy Predictor

### An End-to-End MLOps Project for Short-Term Electricity Demand Forecasting

Smart Energy Predictor is a machine learning-based application designed to forecast electricity demand in Delhi **5 minutes ahead** using historical electricity consumption and weather-related features.

The project demonstrates a complete MLOps workflow, including data preprocessing, feature engineering, model training, experiment tracking, model evaluation, API development, Docker containerization, automated testing, CI and cloud deployment.

## 🚀 Live Application

**Live Demo:** [Smart Energy Predictor](YOUR_RENDER_URL)

**GitHub Repository:** [Smart Energy Predictor](https://github.com/Twarit01/Smart-Energy-Predictor)

## ✨ Key Features

- **Electricity Demand Forecasting:** Predicts electricity demand 5 minutes ahead.
- **Machine Learning:** Trains and compares Linear Regression and Random Forest models.
- **Feature Engineering:** Uses temporal, lag, rolling and weather-related features.
- **Model Evaluation:** Evaluates models using MAE, RMSE and R².
- **MLflow Tracking:** Tracks experiments and model performance.
- **Interactive Dashboard:** Displays predictions, analytics and model performance.
- **Prediction History:** Maintains prediction records and supports CSV export.
- **REST API:** Provides prediction and model information through FastAPI.
- **Automated Testing:** Uses Pytest to validate application functionality.
- **Continuous Integration:** GitHub Actions automates training and testing checks.
- **Docker Containerization:** Packages the application and its dependencies.
- **Cloud Deployment:** Deployed using Render.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Backend | FastAPI |
| Experiment Tracking | MLflow |
| Model Serialization | Joblib |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Deployment | Render |

## 🏗️ Project Architecture

```text
                Historical Dataset
                        |
                        v
                Data Preprocessing
                        |
                        v
                 Feature Engineering
                        |
                        v
                  Model Training
                        |
               -------------------
               |                 |
               v                 v
         Linear Regression   Random Forest
               |                 |
               -------------------
                        |
                        v
                  Model Evaluation
                        |
                        v
                  MLflow Tracking
                        |
                        v
                  Model Selection
                        |
                        v
                 Model Serialization
                     (Joblib)
                        |
                        v
                  FastAPI Backend
                        |
              ---------------------
              |                   |
              v                   v
         REST API             Dashboard
              |                   |
              ---------------------
                        |
                        v
                 Docker Container
                        |
                        v
                  Render Deployment
```

## 🤖 Machine Learning Models

The project trains and compares two regression algorithms:

1. Linear Regression
2. Random Forest Regressor

### Model Evaluation Results

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 23.6682 | 26.25 |
| RMSE | 37.0274 | 49.68 |
| R² Score | 0.99924 | 0.99863 |

**Selected Model: Linear Regression**

Linear Regression achieved lower MAE and RMSE and a higher R² score on the evaluation dataset.

## 📊 Dashboard Modules

The interactive dashboard includes:

- **Overview:** Summary of the forecasting application.
- **Predictions:** Generate electricity demand forecasts.
- **Analytics:** Explore electricity demand analytics.
- **Model Performance:** Compare trained models and evaluation metrics.
- **Prediction History:** Review previous predictions and export records.

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the dashboard |
| `/health` | GET | Application health check |
| `/predict` | POST | Generates a manual prediction |
| `/demo-predict` | GET | Generates a demonstration prediction |
| `/model-metrics` | GET | Retrieves MLflow model metrics |
| `/prediction-history` | GET | Retrieves prediction history |
| `/docs` | GET | Interactive API documentation |

## 📁 Project Structure

```text
Smart-Energy-Predictor/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── ML_Dataset.csv
│
├── model/
│   └── model.joblib
│
├── static/
│   └── index.html
│
├── tests/
│   └── test_app.py
│
├── logs/
├── notebooks/
│
├── app.py
├── train.py
├── requirements.txt
├── requirements-serve.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── mlflow.db
└── README.md
```

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Twarit01/Smart-Energy-Predictor.git
cd Smart-Energy-Predictor
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
uvicorn app:app --reload
```

Open the application at:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

## 🐳 Docker Setup

Build the Docker image:

```bash
docker build -t smart-energy-predictor .
```

Run the container:

```bash
docker run -d \
  --name energy-predictor \
  -p 8000:8000 \
  smart-energy-predictor
```

Check running containers:

```bash
docker ps
```

View container logs:

```bash
docker logs energy-predictor
```

Stop the container:

```bash
docker stop energy-predictor
```

## 📈 MLflow Experiment Tracking

MLflow is used to track and compare machine learning experiments.

Tracked metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Run the MLflow UI locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open:

```text
http://localhost:5000
```

## 🧪 Testing

The project uses Pytest for automated testing.

Run:

```bash
pytest -v
```

GitHub Actions automatically executes the configured CI workflow.

## ☁️ Deployment

The application is deployed on **Render** using Docker.

Deployment workflow:

1. Source code is pushed to GitHub.
2. Render retrieves the latest repository changes.
3. Render builds the Docker image.
4. The container starts the FastAPI application.
5. The application becomes accessible through a public URL.

## 🔮 Future Improvements

- Persistent cloud storage for prediction history.
- Automated model retraining.
- Model versioning and lifecycle management.
- Prediction performance monitoring.
- Improved forecasting models.

## 📄 License

Developed for academic and educational purposes.