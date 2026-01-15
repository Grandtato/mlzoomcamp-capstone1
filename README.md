# Breast Cancer Classification - Capstone 1

## Problem Description
This project builds a machine learning model to predict whether a breast mass is **malignant** (harmful) or **benign** (non-harmful) based on digital image measurements (radius, texture, smoothness, etc.).

Early diagnosis of breast cancer significantly improves survival rates. This service can assist medical professionals by providing an automated second opinion based on the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

## Project Structure
* `notebook.ipynb`: Exploratory Data Analysis (EDA) and Model Comparison (Logistic Regression vs Random Forest).
* `train.py`: Exports the best model (Logistic Regression) to a file.
* `app.py`: Flask web service for predictions.
* `Dockerfile`: Container configuration.

## Setup & Usage
This project trains a binary classifier and serves predictions via a REST API.

## Setup (WSL2)
```bash
source ~/.venvs/mlzoomcamp-2025/bin/activate
pip install -r requirements.txt


# train model (creates models/model.joblib and models/example_request.json if your script does that)
python train.py

#Artifacts created:

#models/model.joblib

#models/example_request.json

# run service
#Run locally (Flask dev server)
python app.py

#TEST
curl -s http://127.0.0.1:9696/health
curl -s -X POST -H "Content-Type: application/json" \
  -d @models/example_request.json \
  http://127.0.0.1:9696/predict

#Run with Docker (Gunicorn)

docker build -t mlzoomcamp-capstone1 .
docker run --rm -p 9696:9696 mlzoomcamp-capstone1

#TEST
curl -s http://127.0.0.1:9696/health
curl -s -X POST -H "Content-Type: application/json" \
  -d @models/example_request.json \
  http://127.0.0.1:9696/predict
