"""
Test script to verify the API is working correctly
"""
import requests
import json


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_status():
    """Test status endpoint"""
    print("\nTesting /api/status endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_inference():
    """Test inference endpoint"""
    print("\nTesting /api/inference endpoint...")
    try:
        data = {
            "latitude": 0.3476,
            "longitude": 32.5825
        }
        response = requests.post(
            "http://localhost:8000/api/inference",
            json=data
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("API Test Suite")
    print("=" * 60)
    print("\nMake sure the server is running on http://localhost:8000\n")
    
    results = {
        "Health Check": test_health(),
        "Status Check": test_status(),
        "Inference": test_inference()
    }
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)


if __name__ == "__main__":
    main()
