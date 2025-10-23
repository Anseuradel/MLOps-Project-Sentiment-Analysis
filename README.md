# MLOps Project -- End-to-End Sentiment Analysis Pipeline

--------------------------------------------

## Project Overview
This project demonstrates a complete MLOps workflow for a semtiment analysis model, covering every step from data ingestion to deployment and monitoring.
It uses a modular design to ensure scalability, reproducibility, and observability of machine learning system in production.

The system includes : 
- Model Training & Evaluzation using Putorch
- FastAPI backend for model serving
- Streamlit dashboard for visualization and predictions
- SQLite databse for prediction logging
- Docker Compose orchestration for local deployment
- Prometheus & Grafana for metrics and monitoring

## Architecture Overview : 

Sentiment-Analysis-project/
├── Dataset/
	└── text.csv                    # Dataset containing reviews from top 10 most popular play store apps
├── k8s/                    
│   ├── configmap.yaml                      
│   │── deployment.yaml 
│   │── grafana.yaml
│   │── hpa.yaml
│   │── persistence.yaml
│   │── prometheus.yaml
│   └── service.yaml
├── output/
│   ├── training_evaluation/                      # Evaluation outputs and plots
│	│   ├── evaluation/
│	│   └── training/
├── src/
│   ├── api/ 
│	│   ├── main.py
│   ├── model/
│	│   ├── data_extraction.py              # Loads raw data from files 
│	│   ├── data_processing.py              # Cleans and tokenizes text data, splits dataset
│	│   ├── evaluate.py                     # Contains evaluation and plotting functions
│	│   ├── dataloader.py                   # Constructs PyTorch DataLoaders
│	│   ├── model.py                        # Defines the SentimentClassifier model architecture
│	│   ├── trainer.py                      # Contains training routines and plotting functions
│   └── inference.py                        # Provides sentiment prediction for new inputs
├── tests/
│   ├── evaluation/
│   │   └── evaluate_model.py               # Function used by github workflow for model evaluation.
├── .gitattributes                            
├── Dockerfile
├── LICENSE
├── launch_k8s.ps1
├── config.py                               # Configuration settings (paths, model parameters, etc.)
├── README.md                               # Project documentation (this file)
└── requirements.txt                        # Dependencies required to run the project

how to start project :
1) launch docker desktop
2) launch terminal -> minikube start -> minikube service ml-service --url
3) - launch promotheus (only scrapes ml-service) -> kubectl port-forward svc/prometheus 9090 -n default
   - launch prometheus stack (only for api metrics (cpu usage etc) -> kubectl port-forward svc/kube-prometheus-stack-prometheus -n monitoring 9090:9090
5) - launch grafana (only for ml-service data) -> kubectl port-forward -n monitoring deployment/grafana 3000 -> go http://localhost:3000/login (admin)
   - launch grafana stack ( only for api metrics (cpu usage etc) -> kubectl port-forward svc/monitoring-grafana -n monitoring 3000:80
7) launch service -> kubectl port-forward svc/ml-service 8000:8000 -n default
8) launch fast api server -> uvicorn src.api.main:app --reload --port 8001
9) Make predictions -> Invoke-RestMethod -Uri "http://localhost:8000/predict" `
>>   -Method Post `
>>   -Headers @{"Content-Type"="application/json"} `
>>   -Body '{"text":"This is a test prediction"}'

8) if i have to make a change in api or model : docker build -t adelanseur95/ml-service:latest .
9) don't forget if model is changed, you need to run locally the file to create the model.joblib
10) Build & push the new image with the model


    docker build -t adelanseur95/ml-service:latest .
    
    docker push adelanseur95/ml-service:latest
    
    
    kubectl set image deployment/ml-service ml-service=adelanseur95/ml-service:latest -n default
    
    kubectl get pods -w
    
    kubectl port-forward svc/ml-service 8000:8000 -n default
    
    Invoke-RestMethod -Uri "http://localhost:8000/predict" `
      -Method Post `
      -Headers @{"Content-Type"="application/json"} `
      -Body '{"text":"This is a test prediction"}'
