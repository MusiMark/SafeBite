# 🚀 Quick Start Guide - Air Quality AI API

Welcome! This guide will get you up and running in 5 minutes.

## ⚡ Super Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
cd d:\AI_project\final2\ai-model-web-app
pip install -r requirements.txt
```

### Step 2: Start the Server
```bash
python run.py
```

### Step 3: Test It!
Open in browser: **http://localhost:8000**

Done! 🎉

---

## 📚 What You Just Built

You now have a **fully functional REST API** with:
- ✅ Real-time air quality analysis
- ✅ 30-minute forecasting
- ✅ Anomaly detection
- ✅ Beautiful web interface
- ✅ Auto-generated API documentation

---

## 🌐 Deployment Options (Choose One)

### Option 1: Render (Easiest - FREE)
1. Go to [render.com](https://render.com)
2. Connect GitHub repo
3. Deploy in 1 click
4. **Time: 5 minutes**

### Option 2: Railway (Fast - $5/mo)
1. Go to [railway.app](https://railway.app)
2. "New Project" → Import from GitHub
3. Deploy automatically
4. **Time: 3 minutes**

### Option 3: Google Cloud Run (Scalable)
```bash
gcloud run deploy --source .
```
**Time: 10 minutes**

### Option 4: Docker (Anywhere)
```bash
docker-compose up -d
```
**Time: 5 minutes**

**Full deployment guides:** See [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **[README.md](README.md)** | Complete project documentation |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Detailed deployment guides for all platforms |
| **[TESTING.md](TESTING.md)** | How to test the API (6 different methods) |
| **[API Docs](http://localhost:8000/docs)** | Interactive API documentation (when server running) |

---

## 🔗 Important URLs (When Running)

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Status**: http://localhost:8000/api/status

---

## 💡 Quick API Test

### Using Browser (Easiest)
1. Open http://localhost:8000
2. Enter coordinates: `0.3476, 32.5825`
3. Click "Analyze"

### Using cURL (Command Line)
```bash
curl -X POST http://localhost:8000/api/inference ^
  -H "Content-Type: application/json" ^
  -d "{\"latitude\": 0.3476, \"longitude\": 32.5825}"
```

### Using Python
```python
import requests
response = requests.post(
    "http://localhost:8000/api/inference",
    json={"latitude": 0.3476, "longitude": 32.5825}
)
print(response.json())
```

---

## 🎯 Next Steps

### For Development
- [ ] Customize the web interface (`src/static/`)
- [ ] Add new endpoints (`src/routers/`)
- [ ] Integrate real data sources
- [ ] Add user authentication

### For Production
- [ ] Choose deployment platform (see DEPLOYMENT.md)
- [ ] Set up environment variables
- [ ] Configure database
- [ ] Set up monitoring
- [ ] Add HTTPS/SSL

### For Integration
- [ ] Share API with team
- [ ] Build mobile app using API
- [ ] Create dashboard
- [ ] Set up webhooks

---

## 🆘 Common Issues

**Problem: Port 8000 already in use**
```bash
# Solution: Use different port
set PORT=8080
python run.py
```

**Problem: Module not found**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**Problem: Model files not found**
- Check that parent directories exist
- Verify model paths in `.env`

**Problem: Slow first request**
- Normal! Models load on first request
- Subsequent requests are much faster

---

## 📞 Get Help

- **API Docs**: http://localhost:8000/docs (interactive)
- **Test Guide**: [TESTING.md](TESTING.md)
- **Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Full Docs**: [README.md](README.md)

---

## 🎉 You're All Set!

Your API is ready to:
1. ✅ Run locally for development
2. ✅ Deploy to cloud platforms
3. ✅ Integrate with other applications
4. ✅ Scale for production use

**Choose your next step:**
- 🧪 Test it thoroughly → [TESTING.md](TESTING.md)
- 🚀 Deploy to cloud → [DEPLOYMENT.md](DEPLOYMENT.md)
- 📚 Learn the API → http://localhost:8000/docs
- 🛠️ Customize it → Edit files in `src/`

**Happy coding! 💻**
