# AI Model Web Application

An AI-powered web application for air quality prediction and anomaly detection. This application integrates multiple machine learning models to provide real-time and forecasted air quality metrics, including EatSafe scores and anomaly detection.

## 🌟 Features

- **Real-time Air Quality Analysis**: Get current air quality metrics for any location
- **30-Minute Forecast**: Predict air quality 30 minutes into the future
- **Anomaly Detection**: Identify unusual patterns in air quality data
- **EatSafe Score**: Calculate food safety scores based on air quality
- **REST API**: Easy-to-use API endpoints for integration
- **Interactive Web UI**: Beautiful web interface for non-technical users

## 🏗️ Project Structure

```
ai-model-web-app/
├── src/
│   ├── app.py                      # Main FastAPI application
│   ├── routers/
│   │   └── inference.py            # API endpoints for inference
│   ├── final2/                     # AI models package
│   │   ├── anomaly_detector/       # Anomaly detection model
│   │   ├── inside_air_predictor/   # Air quality prediction model
│   │   ├── safeBite_score/         # EatSafe scoring model
│   │   └── data_processing.py      # Data preprocessing utilities
│   └── static/
│       ├── index.html              # Web interface
│       └── app.js                  # Frontend JavaScript
├── run.py                          # Application runner
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose setup
├── .env.example                    # Environment variables template
├── DEPLOYMENT.md                   # Deployment guide
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-model-web-app
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   copy .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application:**
   ```bash
   python run.py
   ```

6. **Access the application:**
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 📡 API Usage

### Endpoints

#### POST /api/inference
Perform air quality inference for given coordinates.

**Request:**
```json
{
  "latitude": 0.3476,
  "longitude": 32.5825
}
```

**Response:**
```json
{
  "eat_score_now": 75.5,
  "eat_score_future": 72.3,
  "current_anomaly_detected": false,
  "future_anomaly_detected": false,
  "latitude": 0.3476,
  "longitude": 32.5825,
  "message": "Inference completed successfully"
}
```

#### GET /api/status
Check API service status.

**Response:**
```json
{
  "status": "online",
  "service": "Air Quality Inference API",
  "models": {
    "anomaly_detector": "active",
    "air_quality_predictor": "active",
    "eatsafe_scorer": "active"
  }
}
```

### Example Usage

**cURL:**
```bash
curl -X POST "http://localhost:8000/api/inference" \
  -H "Content-Type: application/json" \
  -d '{"latitude": 0.3476, "longitude": 32.5825}'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/inference",
    json={"latitude": 0.3476, "longitude": 32.5825}
)
print(response.json())
```

**JavaScript:**
```javascript
fetch('http://localhost:8000/api/inference', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({latitude: 0.3476, longitude: 32.5825})
})
.then(res => res.json())
.then(data => console.log(data));
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Using Docker
```bash
# Build
docker build -t air-quality-api .

# Run
docker run -p 8000:8000 air-quality-api
```

## ☁️ Cloud Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment guides for:
- ✅ **Render** (Free tier available)
- ✅ **Railway** (Easy deployment)
- ✅ **Google Cloud Run** (Serverless)
- ✅ **AWS Elastic Beanstalk**
- ✅ **Azure App Service**
- ✅ **Heroku**

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# API Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Model paths
ANOMALY_MODEL_PATH=../anomaly_detector/out_train
INSIDE_AIR_MODEL_PATH=../inside_air_predictor/best_model.pth
EATSAFE_MODEL_PATH=../safeBite_score/eatsafe_model.keras

# CORS settings
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
```

## 🧪 Testing

Test the API using the interactive documentation:
```
http://localhost:8000/docs
```

Or use the web interface:
```
http://localhost:8000
```

## 📊 Models

This application uses three main AI models:

1. **Anomaly Detector**: Detects unusual patterns in air quality data using the Tukey method
2. **Inside Air Predictor**: LSTM-based model for forecasting indoor air quality
3. **EatSafe Score**: Neural network model for calculating food safety scores

## 🛠️ Development

### Project Dependencies

Main libraries:
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **TensorFlow**: Deep learning (EatSafe model)
- **PyTorch**: Deep learning (LSTM model)
- **Scikit-learn**: Machine learning utilities

### Adding New Features

1. Create new endpoints in `src/routers/`
2. Add model logic in `src/final2/`
3. Update frontend in `src/static/`
4. Test using `/docs` endpoint

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**Import errors:**
```bash
pip install -r requirements.txt --upgrade
```

**Port already in use:**
```bash
# Change PORT in .env or:
python run.py  # Will use PORT from environment
```

**Model not found:**
- Ensure model files are in the correct directories
- Check paths in `.env`
- Verify model files were copied from parent directory

## 📞 Support

For issues and questions:
- Check the [DEPLOYMENT.md](DEPLOYMENT.md) guide
- Review API docs at `/docs`
- Check application logs
- Open an issue on GitHub

---

**Made with ❤️ for better air quality monitoring**