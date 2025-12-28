# ML Zoomcamp 2025 - Capstone 1

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
