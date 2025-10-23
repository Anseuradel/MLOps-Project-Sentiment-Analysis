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

```Python
Dataset/
    └── text.txt
outputs/
    └── training_evaluation/
        ├── evaluation/
            └── run_28-09-2025-16-16-25/
                ├── classification_report.png
                ├── confidence_histogram.png
                ├── confusion_matrix.png
                └── metrics.txt
        └── training/
            └── run_28-09-2025-14-16-46/
                ├── accuracy_and_loss_plot.png
                └── training_history.json
src/
    ├── api/
        ├── api.py
        ├── database.py
        └── main.py
    ├── app/
        └── streamlit_app.py
    ├── model/
        ├── __init__.py
        ├── data_extraction.py
        ├── data_processing.py
        ├── dataloader.py
        ├── evaluate.py
        ├── inference.py
        ├── main_loading_by_chunks.py
        ├── main2.py
        ├── model.py
        ├── run_main_colab_git_lfs.py
        ├── run_main_colab_hugg.py
        └── trainer.py
    └── __init__.py
tests/
    ├── load_test.ps1
    ├── test_api.py
    └── test_model.py
.gitattributes
.gitignore
config.py
docker-compose.yaml
Dockerfile
last_chunk.txt
launch_k8s.ps1
LICENSE
README.md
requirements.txt
```

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
