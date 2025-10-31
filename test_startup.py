"""
Test script to verify the app starts quickly without loading ML models
Run this before deploying to Render
"""
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("🧪 TESTING FAST STARTUP (NO ML MODELS)")
print("=" * 60)

start_time = time.time()

print("\n1️⃣ Importing FastAPI app...")
from app import app

import_time = time.time() - start_time
print(f"   ✅ Import took: {import_time:.2f} seconds")

if import_time > 10:
    print("   ⚠️  WARNING: Import took more than 10 seconds!")
    print("   This may cause Render timeout issues.")
else:
    print("   ✅ Import time is acceptable for Render deployment")

print("\n2️⃣ Checking health endpoint...")
from fastapi.testclient import TestClient
client = TestClient(app)

health_start = time.time()
response = client.get("/health")
health_time = time.time() - health_start

print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}")
print(f"   ✅ Health check took: {health_time:.2f} seconds")

print("\n3️⃣ Checking main page...")
response = client.get("/")
print(f"   Status: {response.status_code}")
print(f"   ✅ Main page accessible")

total_time = time.time() - start_time

print("\n" + "=" * 60)
print(f"🎉 TOTAL STARTUP TIME: {total_time:.2f} seconds")
print("=" * 60)

if total_time < 15:
    print("✅ EXCELLENT! App starts fast enough for Render")
    print("✅ Port will bind within timeout period")
    sys.exit(0)
elif total_time < 30:
    print("⚠️  ACCEPTABLE, but close to timeout")
    print("⚠️  May work on Render, but monitor closely")
    sys.exit(0)
else:
    print("❌ TOO SLOW! App will timeout on Render")
    print("❌ Startup must be under 30 seconds")
    sys.exit(1)
