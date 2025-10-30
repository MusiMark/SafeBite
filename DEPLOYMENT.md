# Deployment Guide

This guide covers different ways to deploy your AI Model API.

## Table of Contents
- [Local Development](#local-development)
- [Cloud Platforms](#cloud-platforms)
  - [Render](#render)
  - [Railway](#railway)
  - [Google Cloud Run](#google-cloud-run)
  - [AWS Elastic Beanstalk](#aws-elastic-beanstalk)
  - [Azure App Service](#azure-app-service)
  - [Heroku](#heroku)
- [Docker Deployment](#docker-deployment)
- [Platform Recommendations](#platform-recommendations)

---

## Local Development

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment file
copy .env.example .env

# 3. Run the server
python run.py
```

The API will be available at `http://localhost:8000`

---

## Cloud Platforms

### 🚀 Render (Recommended - Free Tier Available)

**Pros:** Free tier, easy setup, automatic deployments from GitHub
**Best for:** Quick demos and prototypes

1. **Create account** at [render.com](https://render.com)
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repository**
4. **Configure:**
   - Name: `air-quality-api`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`
5. **Add Environment Variables** from `.env.example`
6. **Deploy!**

**Note:** Free tier spins down after inactivity. Upgrade to paid for always-on.

---

### 🚂 Railway (Easy & Fast)

**Pros:** Simple deployment, generous free tier, excellent DX
**Best for:** MVPs and small projects

1. **Visit** [railway.app](https://railway.app)
2. **Click "Start a New Project"**
3. **Deploy from GitHub repo**
4. Railway auto-detects Python and creates service
5. **Add environment variables**
6. **Deploy automatically**

Cost: ~$5/month after free tier

---

### ☁️ Google Cloud Run (Serverless)

**Pros:** Pay only for what you use, scales to zero, production-ready
**Best for:** Production apps with variable traffic

```bash
# 1. Install Google Cloud SDK
# 2. Authenticate
gcloud auth login

# 3. Build and deploy
gcloud run deploy air-quality-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

Cost: Free tier includes 2M requests/month

---

### 📦 AWS Elastic Beanstalk

**Pros:** Full AWS integration, auto-scaling, mature platform
**Best for:** Enterprise applications

```bash
# 1. Install EB CLI
pip install awsebcli

# 2. Initialize
eb init -p python-3.10 air-quality-api

# 3. Create environment
eb create air-quality-api-env

# 4. Deploy
eb deploy
```

Cost: ~$25/month minimum (EC2 instance)

---

### 🌐 Azure App Service

**Pros:** Good Windows integration, enterprise features
**Best for:** Microsoft ecosystem projects

```bash
# 1. Install Azure CLI
# 2. Login
az login

# 3. Create app
az webapp up --name air-quality-api \
  --runtime "PYTHON:3.10" \
  --sku B1

# 4. Configure startup
az webapp config set --startup-file "run.py"
```

Cost: ~$13/month (B1 tier)

---

### 💜 Heroku (Classic Choice)

**Pros:** Easy to use, lots of add-ons
**Note:** No more free tier as of 2022

1. **Create** `Procfile`:
```
web: uvicorn src.app:app --host 0.0.0.0 --port $PORT
```

2. **Deploy:**
```bash
heroku login
heroku create air-quality-api
git push heroku main
```

Cost: $7/month minimum (Eco dyno)

---

## Docker Deployment

### Build and Run Locally
```bash
# Build
docker build -t air-quality-api .

# Run
docker run -p 8000:8000 air-quality-api
```

### Using Docker Compose
```bash
docker-compose up -d
```

### Deploy to any Docker-compatible platform:
- **DigitalOcean App Platform**
- **Fly.io**
- **Google Cloud Run** (from container)
- **AWS ECS/Fargate**

---

## Platform Recommendations

| Use Case | Platform | Cost | Difficulty |
|----------|----------|------|------------|
| **Demo/Prototype** | Render | Free | ⭐ Easy |
| **MVP** | Railway | $5/mo | ⭐ Easy |
| **Small Production** | Google Cloud Run | Pay-per-use | ⭐⭐ Medium |
| **Scalable App** | AWS/Azure | $25+/mo | ⭐⭐⭐ Hard |
| **Enterprise** | AWS/Azure/GCP | $100+/mo | ⭐⭐⭐ Hard |

---

## Important Notes

### Model Files
⚠️ **Large model files (>500MB) may cause issues on some platforms.**

Solutions:
1. **Use cloud storage** (S3, Google Cloud Storage)
2. **Download models on startup**
3. **Use model compression**
4. **Consider model serving platforms** (TensorFlow Serving, TorchServe)

### Environment Variables
Always set these in your deployment platform:
```
HOST=0.0.0.0
PORT=8000 (or $PORT for cloud platforms)
DEBUG=False
```

### Database
Currently uses CSV files. For production:
1. Set up PostgreSQL or MongoDB
2. Use cloud database services (AWS RDS, Google Cloud SQL)
3. Update connection strings in `.env`

---

## Quick Deployment Checklist

- [ ] Update `.env` with production values
- [ ] Set `DEBUG=False`
- [ ] Add your domain to `ALLOWED_ORIGINS`
- [ ] Set up database (if needed)
- [ ] Upload model files or configure cloud storage
- [ ] Set up monitoring/logging
- [ ] Configure HTTPS/SSL
- [ ] Set up CI/CD (optional)
- [ ] Test all endpoints
- [ ] Set up backups

---

## Support

For deployment issues:
- Check platform documentation
- Review application logs
- Test locally with `DEBUG=True`
- Ensure all dependencies are in `requirements.txt`

Good luck with your deployment! 🚀
