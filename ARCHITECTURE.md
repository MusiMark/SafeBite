# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Web Browser  │  │ Mobile App   │  │ API Client   │       │
│  │ (HTML/JS)    │  │ (iOS/Android)│  │ (Python/etc) │       │
│  └──────┬───────┘  └──────┬───────┘  └───────┬──────┘       │
│         │                 │                  │              │
│         └─────────────────┴──────────────────┘              │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │ HTTP/HTTPS
                            │
┌───────────────────────────┼─────────────────────────────────┐
│                    API LAYER (FastAPI)                      │
├───────────────────────────┼─────────────────────────────────┤
│                           │                                 │
│  ┌────────────────────────▼──────────────────────────┐      │
│  │         app.py (Main Application)                 │      │
│  │  • CORS Middleware                                │      │
│  │  • Static Files Serving                           │      │
│  │  • Route Registration                             │      │
│  └────────────────────────┬──────────────────────────┘      │
│                           │                                 │
│  ┌────────────────────────▼──────────────────────────┐      │
│  │   routers/inference.py (API Endpoints)            │      │
│  │  • POST /api/inference                            │      │
│  │  • GET  /api/status                               │      │
│  │  • Request Validation (Pydantic)                  │      │
│  └────────────────────────┬──────────────────────────┘      │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────┐
│                   SERVICE LAYER                             │
├───────────────────────────┼─────────────────────────────────┤
│                           │                                 │
│  ┌────────────────────────▼──────────────────────────┐      │
│  │    data_processing.py                             │      │
│  │  • Data Preprocessing                             │      │
│  │  • Feature Engineering                            │      │
│  │  • Time Series Processing                         │      │
│  └────────────────────────┬──────────────────────────┘      │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            │
                            ▼ 
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODEL LAYER                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   ┌─────────────┐ │
│  │   Anomaly    │  │  Inside Air  │  │   EatSafe    │   │   PM2.5     │ │
│  │   Detector   │  │  Predictor   │  │    Score     │   │  Predictor  │ │
│  │    (GAT)     │  │   (LSTM)     │  │  ( Neural    │   │  (Graph     │ │
│  └──────────────┘  └──────────────┘  │   Network)   │   │Convolutional│ │
│                                      │              │   │  Network)   │ │ 
│                                      └──────────────┘   └─────────────┘ │ 
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
User Input (Lat, Long)
          │
          ▼
┌─────────────────────┐
│ API Validation      │
│ (Pydantic)          │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Load Historical     │
│ Data (CSV)          │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Generate Future     │
│ Predictions (LSTM)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Preprocess Data     │
│ (Feature Eng.)      │
└─────────┬───────────┘
          │
          ├─────────────────┬─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Anomaly         │ │ Air Quality     │ │ EatSafe         │
│ Detection       │ │ Prediction      │ │ Scoring         │
└─────────┬───────┘ └───────┬─────────┘ └─────────┬───────┘
          │                 │                     │
          └─────────────────┼─────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Aggregate       │
                   │ Results         │
                   └────────┬────────┘
                            │
                            │
                            ▼
                   ┌─────────────────┐
                   │ JSON Response   │
                   └─────────────────┘
```

## Request/Response Flow

```
┌──────────┐                                          ┌──────────┐
│  Client  │                                          │  Server  │
└────┬─────┘                                          └────┬─────┘
     │                                                     │
     │  POST /api/inference                                │
     │  {lat: 0.3476, lon: 32.5825}                        │
     ├────────────────────────────────────────────────────►│
     │                                                      │
     │                                         Validate Input
     │                                                      │
     │                                         Load Data    │
     │                                                      │
     │                                         Preprocess   │
     │                                                      │
     │                                         Run Models   │
     │                                         ┌──────┐     │
     │                                         │LSTM  │     │
     │                                         └──────┘     │
     │                                         ┌──────┐     │
     │                                         │Anomaly│    │
     │                                         └──────┘     │
     │                                         ┌──────┐     │
     │                                         │Score │     │
     │                                         └──────┘     │
     │                                                      │
     │  200 OK                                              │
     │  {                                                   │
     │    "eat_score_now": 75.5,                           │
     │    "eat_score_future": 72.3,                        │
     │    "current_anomaly_detected": false,               │
     │    "future_anomaly_detected": false                 │
     │  }                                                   │
     │◄────────────────────────────────────────────────────┤
     │                                                      │
```

## Deployment Architecture

### Local Development
```
┌─────────────────────────────────────┐
│      Development Machine            │
│                                     │
│  ┌───────────────────────────┐      │
│  │  Python 3.10+             │      │
│  │  └─ Uvicorn Server        │      │
│  │     └─ FastAPI App        │      │
│  │        └─ AI Models       │      │
│  └───────────────────────────┘      │
│                                     │
│  Access: localhost:8000             │
└─────────────────────────────────────┘
```

### Cloud Deployment (Example: Render)
```
┌────────────────────────────────────────────┐
│           Internet Users                   │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│       Load Balancer (HTTPS)                 │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│       Render Web Service                    │
│                                             │
│  ┌─────────────────────────────────┐       │
│  │  Docker Container                │       │
│  │  ┌───────────────────────────┐  │       │
│  │  │  Gunicorn/Uvicorn         │  │       │
│  │  │  └─ FastAPI Application   │  │       │
│  │  │     └─ AI Models          │  │       │
│  │  └───────────────────────────┘  │       │
│  └─────────────────────────────────┘       │
│                                             │
│  Auto-scaling, Health Checks, Logs          │
└─────────────────────────────────────────────┘
```

### Docker Deployment
```
┌─────────────────────────────────────────────┐
│       Docker Host (Any Platform)            │
│                                             │
│  ┌─────────────────────────────────┐       │
│  │  Docker Container (api)          │       │
│  │  ┌───────────────────────────┐  │       │
│  │  │  Python 3.10 Slim          │  │       │
│  │  │  ├─ FastAPI App            │  │       │
│  │  │  ├─ Models                 │  │       │
│  │  │  └─ Dependencies           │  │       │
│  │  └───────────────────────────┘  │       │
│  │  Port: 8000 → Host             │       │
│  └─────────────────────────────────┘       │
│                                             │
└─────────────────────────────────────────────┘
```

## Technology Stack

```
┌─────────────────────────────────────────────┐
│            Frontend Layer                   │
├─────────────────────────────────────────────┤
│  • HTML5, CSS3                              │
│  • Vanilla JavaScript (ES6+)               │
│  • Fetch API for HTTP requests              │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│            API Framework                    │
├─────────────────────────────────────────────┤
│  • FastAPI (Python web framework)          │
│  • Uvicorn (ASGI server)                   │
│  • Pydantic (Data validation)              │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│            Data Processing                  │
├─────────────────────────────────────────────┤
│  • Pandas (Data manipulation)              │
│  • NumPy (Numerical computing)             │
│  • Scikit-learn (ML utilities)             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│            AI/ML Models                     │
├─────────────────────────────────────────────┤
│  • PyTorch (LSTM model)                    │
│  • TensorFlow/Keras (EatSafe model)        │
│  • Statistical methods (Anomaly detection) │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│            Deployment                       │
├─────────────────────────────────────────────┤
│  • Docker (Containerization)               │
│  • Cloud platforms (Render, Railway, etc.) │
│  • CI/CD (GitHub Actions ready)            │
└─────────────────────────────────────────────┘
```

## Security Considerations

```
┌─────────────────────────────────────────────┐
│         Security Layer                      │
├─────────────────────────────────────────────┤
│                                             │
│  ✓ CORS Configuration (Middleware)          │
│  ✓ Input Validation (Pydantic)              │
│  ✓ Rate Limiting (TODO)                     │
│  ✓ API Authentication (TODO)                │
│  ✓ HTTPS/SSL (Production)                   │
│  ✓ Environment Variables (.env)             │
│                                             │
└─────────────────────────────────────────────┘
```

## Scalability Options

### Horizontal Scaling
```
          ┌──────────┐
          │  Client  │
          └────┬─────┘
               │
         ┌─────▼──────┐
         │   Load     │
         │  Balancer  │
         └─────┬──────┘
               │
       ┌───────┼───────┐
       │       │       │
    ┌──▼─┐  ┌──▼─┐  ┌──▼─┐
    │API │  │API │  │API │
    │ 1  │  │ 2  │  │ 3  │
    └────┘  └────┘  └────┘
```

### With Caching
```
  ┌──────────┐
  │  Client  │
  └────┬─────┘
       │
  ┌────▼─────┐
  │  Redis   │  ← Cache predictions
  │  Cache   │
  └────┬─────┘
       │ (miss)
  ┌────▼─────┐
  │   API    │
  └────┬─────┘
       │
  ┌────▼─────┐
  │  Models  │
  └──────────┘
```

---

This architecture is designed to be:
- **Scalable**: Easy to add more instances
- **Maintainable**: Clear separation of concerns
- **Flexible**: Can swap models without changing API
- **Production-ready**: Health checks, error handling, logging
