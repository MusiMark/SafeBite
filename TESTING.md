# API Testing Guide

This guide helps you test the Air Quality AI API.

## Quick Start

### 1. Start the Server

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Or manually:**
```bash
python run.py
```

### 2. Verify Server is Running

Open your browser and visit:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## Testing Methods

### Method 1: Web Interface (Easiest)

1. Open http://localhost:8000
2. Enter coordinates:
   - Latitude: `0.3476`
   - Longitude: `32.5825`
3. Click "Analyze"
4. View results

### Method 2: Interactive API Docs

1. Open http://localhost:8000/docs
2. Click on `/api/inference` endpoint
3. Click "Try it out"
4. Enter test data:
   ```json
   {
     "latitude": 0.3476,
     "longitude": 32.5825
   }
   ```
5. Click "Execute"
6. View response

### Method 3: Python Script

Run the test script:
```bash
python test_api.py
```

### Method 4: cURL (Command Line)

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Status Check:**
```bash
curl http://localhost:8000/api/status
```

**Inference:**
```bash
curl -X POST "http://localhost:8000/api/inference" ^
  -H "Content-Type: application/json" ^
  -d "{\"latitude\": 0.3476, \"longitude\": 32.5825}"
```

### Method 5: Python Requests

```python
import requests

# Test inference
response = requests.post(
    "http://localhost:8000/api/inference",
    json={"latitude": 0.3476, "longitude": 32.5825}
)
print(response.json())
```

### Method 6: Postman

1. Download Postman
2. Create new POST request
3. URL: `http://localhost:8000/api/inference`
4. Headers: `Content-Type: application/json`
5. Body (raw JSON):
   ```json
   {
     "latitude": 0.3476,
     "longitude": 32.5825
   }
   ```
6. Send request

---

## Sample Test Data

### Valid Coordinates

**Kampala, Uganda:**
```json
{
  "latitude": 0.3476,
  "longitude": 32.5825
}
```

**Nairobi, Kenya:**
```json
{
  "latitude": -1.2921,
  "longitude": 36.8219
}
```

**New York, USA:**
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060
}
```

**Tokyo, Japan:**
```json
{
  "latitude": 35.6762,
  "longitude": 139.6503
}
```

### Invalid Coordinates (Should Return Error)

**Invalid latitude:**
```json
{
  "latitude": 95.0,
  "longitude": 32.5825
}
```

**Invalid longitude:**
```json
{
  "latitude": 0.3476,
  "longitude": 200.0
}
```

**Missing data:**
```json
{
  "latitude": 0.3476
}
```

---

## Expected Responses

### Successful Response

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

### Error Response

```json
{
  "detail": "Latitude must be between -90 and 90"
}
```

---

## Performance Testing

### Load Testing with Apache Bench

```bash
# 100 requests, 10 concurrent
ab -n 100 -c 10 -p test_data.json -T application/json http://localhost:8000/api/inference
```

Create `test_data.json`:
```json
{"latitude": 0.3476, "longitude": 32.5825}
```

### Load Testing with Python

```python
import requests
import concurrent.futures
import time

def make_request():
    return requests.post(
        "http://localhost:8000/api/inference",
        json={"latitude": 0.3476, "longitude": 32.5825}
    )

# Send 50 concurrent requests
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(make_request) for _ in range(50)]
    results = [f.result() for f in futures]

elapsed = time.time() - start
success = sum(1 for r in results if r.status_code == 200)
print(f"Completed {len(results)} requests in {elapsed:.2f}s")
print(f"Success rate: {success}/{len(results)}")
```

---

## Troubleshooting

### Server Won't Start

**Check port availability:**
```bash
netstat -ano | findstr :8000
```

**Try different port:**
```bash
set PORT=8080
python run.py
```

### Connection Refused

- Make sure server is running
- Check firewall settings
- Verify correct URL (http://localhost:8000)

### Slow Response

- Large model files may cause delays on first request
- Subsequent requests should be faster (models cached)
- Consider using smaller models or model optimization

### Import Errors

```bash
pip install -r requirements.txt --upgrade
```

### Model Not Found

- Check model files exist in parent directories
- Verify paths in `.env`
- Check file permissions

---

## Monitoring

### Check Logs

Server logs show all requests:
```
INFO:     127.0.0.1:52154 - "POST /api/inference HTTP/1.1" 200 OK
```

### Response Times

Look for response time in logs or use:
```python
import time
start = time.time()
response = requests.post(...)
print(f"Response time: {time.time() - start:.2f}s")
```

---

## Integration Testing

### Example Integration Test

```python
import unittest
import requests

class TestAPI(unittest.TestCase):
    BASE_URL = "http://localhost:8000"
    
    def test_health(self):
        response = requests.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
    
    def test_inference_valid(self):
        response = requests.post(
            f"{self.BASE_URL}/api/inference",
            json={"latitude": 0.3476, "longitude": 32.5825}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("eat_score_now", data)
        self.assertIn("eat_score_future", data)
    
    def test_inference_invalid_latitude(self):
        response = requests.post(
            f"{self.BASE_URL}/api/inference",
            json={"latitude": 95.0, "longitude": 32.5825}
        )
        self.assertEqual(response.status_code, 422)

if __name__ == '__main__':
    unittest.main()
```

Run tests:
```bash
python -m unittest test_integration.py
```

---

## Next Steps

After testing locally:
1. Review [DEPLOYMENT.md](DEPLOYMENT.md) for cloud deployment
2. Set up continuous integration
3. Add monitoring and logging
4. Implement authentication if needed
5. Set up database for production data

---

**Happy Testing! 🧪**
