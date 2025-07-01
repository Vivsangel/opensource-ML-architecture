# Open Source ML Platform

A complete, production-ready machine learning platform built with open-source tools, providing alternatives to AWS SageMaker and Google Vertex AI.

## 🚀 Features

### Core Capabilities
- **Real-time Inference**: FastAPI-based REST API for low-latency predictions
- **Batch Processing**: Celery-powered distributed batch inference
- **Model Training**: Automated model training with hyperparameter tracking
- **Model Registry**: MLflow-based model versioning and metadata management
- **Experiment Tracking**: Complete MLOps workflow with experiment management
- **Auto-scaling**: Kubernetes-ready with horizontal pod autoscaling
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Multi-framework Support**: Scikit-learn, PyTorch, TensorFlow

### Production Features
- **High Availability**: Load balancing and health checks
- **Fault Tolerance**: Redis-backed job queuing and retry mechanisms
- **Security**: API authentication and role-based access control ready
- **Observability**: Structured logging and comprehensive metrics
- **Data Pipeline**: Built-in data validation and preprocessing
- **Model Deployment**: Blue-green deployments and A/B testing support

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI/API    │    │   Batch Jobs    │    │   Training      │
│   (FastAPI)     │    │   (Celery)      │    │   (MLflow)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Redis       │    │   PostgreSQL    │    │   Model Store   │
│  (Cache/Queue)  │    │  (Metadata)     │    │   (MLflow)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                     Monitoring Stack                           │
│              (Prometheus + Grafana)                           │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone and setup**:
```bash
git clone <your-repo>
cd ml-platform
```

2. **Start all services**:
```bash
docker-compose up -d
```

3. **Access services**:
- API: http://localhost:8000
- MLflow UI: http://localhost:5000
- Celery Flower: http://localhost:5555
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

### Option 2: Local Development

1. **Install dependencies**:
```bash
chmod +x setup.sh
./setup.sh
```

2. **Start services manually**:
```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Celery worker
source venv/bin/activate
celery -A ml_platform.celery_app worker --loglevel=info

# Terminal 3: Start API server
source venv/bin/activate
uvicorn ml_platform:app --reload
```

### Option 3: Kubernetes Production Deployment

1. **Apply Kubernetes manifests**:
```bash
kubectl apply -f kubernetes/
```

2. **Get service endpoints**:
```bash
kubectl get services -n ml-platform
```

## 🧪 Usage Examples

### 1. Train a Model

```python
import requests

# Train a new model
training_request = {
    "dataset_path": "data/train_data.csv",
    "model_type": "random_forest_classifier",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
    },
    "experiment_name": "customer_churn"
}

response = requests.post("http://localhost:8000/train", json=training_request)
print(response.json())
```

### 2. Deploy Model

```python
# Deploy trained model
model_name = "customer_churn_model"
version = "1"

response = requests.post(f"http://localhost:8000/deploy/{model_name}/{version}")
print(response.json())
```

### 3. Real-time Prediction

```python
# Make real-time prediction
prediction_request = {
    "features": [0.5, -0.3, 1.2, 0.8, -1.1],
    "model_name": "customer_churn_model",
    "model_version": "latest"
}

response = requests.post("http://localhost:8000/predict", json=prediction_request)
print(response.json())
```

### 4. Batch Inference

```python
# Create batch job
batch_request = {
    "input_path": "data/batch_input.csv",
    "output_path": "data/batch_output.csv", 
    "model_name": "customer_churn_model"
}

response = requests.post("http://localhost:8000/batch-inference", json=batch_request)
job_id = response.json()["job_id"]

# Check job status
status_response = requests.get(f"http://localhost:8000/batch-jobs/{job_id}")
print(status_response.json())
```

## 📊 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Real-time prediction |
| POST | `/batch-inference` | Create batch job |
| GET | `/batch-jobs/{job_id}` | Get job status |
| POST | `/train` | Train new model |
| POST | `/deploy/{model}/{version}` | Deploy model |
| GET | `/models` | List all models |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

### Model Training API

```json
{
  "dataset_path": "path/to/training/data.csv",
  "model_type": "random_forest_classifier|random_forest_regressor",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "experiment_name": "experiment_name"
}
```

### Prediction API

```json
{
  "features": [1.0, 2.0, 3.0] or {"feature1": 1.0, "feature2": 2.0},
  "model_name": "model_name", 
  "model_version": "latest|1|2|3"
}
```

## 🔧 Configuration

### Environment Variables

```bash
# Core settings
MODEL_REGISTRY_URI=sqlite:///mlflow.db
REDIS_URL=redis://localhost:6379
DATABASE_URL=sqlite:///platform.db

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# Batch processing
CELERY_BROKER=redis://localhost:6379/0
CELERY_BACKEND=redis://localhost:6379/0

# Monitoring
METRICS_PORT=8001
LOG_LEVEL=INFO
```

### Production Configuration

For production deployments, update the configuration:

```python
# config.py
@dataclass
class ProductionConfig:
    model_registry_uri: str = "postgresql://user:pass@db:5432/mlflow"
    redis_url: str = "redis://redis-cluster:6379"
    database_url: str = "postgresql://user:pass@db:5432/platform"
    
    # Security
    api_key_required: bool = True
    jwt_secret: str = os.getenv("JWT_SECRET")
    
    # Performance
    max_workers: int = 8
    batch_size: int = 1000
    cache_ttl: int = 3600
```

## 📈 Monitoring and Observability

### Metrics Available

- `ml_predictions_total`: Total number of predictions
- `ml_prediction_duration_seconds`: Prediction latency histogram
- `ml_model_accuracy`: Model accuracy gauge
- `ml_batch_jobs_total`: Total batch jobs processed
- `ml_training_jobs_total`: Total training jobs

### Custom Dashboards

The platform includes pre-built Grafana dashboards for:
- Model performance metrics
- API response times
- Resource utilization
- Error rates and alerts

### Logs

Structured logging with correlation IDs:
```json
{
  "timestamp": "2025-06-30T10:30:00Z",
  "level": "INFO",
  "service": "ml-platform-api",
  "correlation_id": "req-123456",
  "model_name": "customer_churn",
  "prediction_time_ms": 45,
  "message": "Prediction completed successfully"
}
```

## 🚀 Advanced Features

### Auto-scaling Configuration

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-platform-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-platform-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### A/B Testing Support

```python
# Deploy multiple model versions
await deploy_model("model_v1", version="1", traffic_weight=80)
await deploy_model("model_v2", version="2", traffic_weight=20)
```

### Model Drift Detection

```python
# Built-in drift detection
drift_detector = ModelDriftDetector(
    reference_data=training_data,
    threshold=0.1
)

# Monitor in real-time
if drift_detector.detect_drift(current_batch):
    alert_manager.send_alert("Model drift detected")
```

## 🔒 Security

### Authentication Setup

```python
# Add to ml_platform.py
from fastapi.security import HTTPBearer
from fastapi import Depends

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # Implement JWT verification
    pass

@app.post("/predict")
async def predict(request: PredictionRequest, token=Depends(verify_token)):
    # Protected endpoint
    pass
```

### Network Security

```yaml
# kubernetes/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-platform-netpol
spec:
  podSelector:
    matchLabels:
      app: ml-platform
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8000
```

## 🧪 Testing

### Run Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Load tests
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

### Example Test

```python
def test_prediction_endpoint():
    response = client.post("/predict", json={
        "features": [1.0, 2.0, 3.0],
        "model_name": "test_model"
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy ML Platform
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Build and test
      run: |
        docker build -t ml-platform:${{ github.sha }} .
        pytest tests/
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/ml-platform-api \
          ml-platform-api=ml-platform:${{ github.sha }}
```

## 📚 Extending the Platform

### Adding New Model Types

```python
class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(**kwargs)
    
    def train(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
```

### Custom Preprocessing

```python
class DataPreprocessor:
    def __init__(self, config):
        self.config = config
    
    def preprocess(self, data):
        # Custom preprocessing logic
        return processed_data
```

## 📋 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Restart Redis
   sudo systemctl restart redis
   ```

2. **Model Loading Issues**
   ```bash
   # Check MLflow tracking server
   curl http://localhost:5000/health
   
   # Verify model artifacts
   ls -la mlruns/
   ```

3. **Celery Worker Not Processing**
   ```bash
   # Check worker status
   celery -A ml_platform.celery_app inspect active
   
   # Restart workers
   celery -A ml_platform.celery_app control shutdown
   ```

### Performance Tuning

```python
# Optimize for high throughput
config = MLPlatformConfig(
    max_workers=multiprocessing.cpu_count() * 2,
    batch_size=1000,
    cache_ttl=3600,
    connection_pool_size=20
)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Documentation: [docs.yourplatform.com](https://docs.yourplatform.com)
- Issues: [GitHub Issues](https://github.com/your-org/ml-platform/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/ml-platform/discussions)
- Slack: [#ml-platform](https://yourorg.slack.com/channels/ml-platform)

---

**Built with ❤️ using open-source tools**

This platform provides enterprise-grade ML capabilities without vendor lock-in, giving you full control over your machine learning infrastructure.