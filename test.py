"""
Test client for BiLSTM prediction API
"""
import requests
import json
from datetime import datetime, timedelta
import random

# API base URL
BASE_URL = "http://localhost:8000"


def generate_sample_data(start_date: str, days: int = 60):
    """Generate sample historical data"""
    start = datetime.strptime(start_date, "%Y-%m-%d")

    historical_data = []
    for i in range(days):
        date = start - timedelta(days=days - i)

        data_point = {
            "branchcode": "GC01",
            "materialcode": "SKU_A",
            "stock_on_hand": random.uniform(400, 600),
            "intransit_qty": random.uniform(50, 150),
            "pending_po_qty": random.uniform(30, 80),
            "lead_time_days": random.randint(5, 10),
            "date": date.strftime("%Y-%m-%d")
        }
        historical_data.append(data_point)

    return historical_data


def test_health_check():
    """Test health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_model_info():
    """Test model info endpoint"""
    print("\n=== Testing Model Info ===")
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Test single prediction"""
    print("\n=== Testing Single Prediction ===")

    # Generate historical data
    historical_data = generate_sample_data("2024-03-15", 60)

    payload = {
        "branchcode": "GC01",
        "materialcode": "SKU_A",
        "historical_data": historical_data
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload
    )

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Sales: {result['predicted_sales']:.2f}")
        print(f"Branch: {result['branchcode']}")
        print(f"Material: {result['materialcode']}")
        print(f"Date: {result['prediction_date']}")
    else:
        print(f"Error: {response.text}")


def test_batch_prediction():
    """Test batch prediction"""
    print("\n=== Testing Batch Prediction ===")

    # Generate data for multiple predictions
    batch_data = []

    for branch in ["GC01", "GC02"]:
        for sku in ["SKU_A", "SKU_B"]:
            historical_data = generate_sample_data("2024-03-15", 60)

            batch_data.append({
                "branchcode": branch,
                "materialcode": sku,
                "historical_data": historical_data
            })

    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json=batch_data
    )

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        for i, pred in enumerate(results["predictions"]):
            print(f"\nPrediction {i + 1}:")
            if "error" not in pred:
                print(f"  Branch: {pred['branchcode']}")
                print(f"  Material: {pred['materialcode']}")
                print(f"  Predicted Sales: {pred['predicted_sales']:.2f}")
            else:
                print(f"  Error: {pred['error']}")
    else:
        print(f"Error: {response.text}")


def test_with_real_sequence():
    """Test with a more realistic sequence"""
    print("\n=== Testing with Realistic Sequence ===")

    # Create a sequence with trend
    base_stock = 500
    historical_data = []

    start = datetime.strptime("2024-03-15", "%Y-%m-%d")

    for i in range(60):
        date = start - timedelta(days=60 - i)

        # Add some trend and noise
        stock = base_stock + (i * 2) + random.uniform(-20, 20)

        data_point = {
            "branchcode": "GC01",
            "materialcode": "SKU_A",
            "stock_on_hand": max(0, stock),
            "intransit_qty": random.uniform(80, 120),
            "pending_po_qty": random.uniform(40, 70),
            "lead_time_days": 7,
            "date": date.strftime("%Y-%m-%d")
        }
        historical_data.append(data_point)

    payload = {
        "branchcode": "GC01",
        "materialcode": "SKU_A",
        "historical_data": historical_data
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload
    )

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Sales: {result['predicted_sales']:.2f}")
        print(f"Last Stock Level: {historical_data[-1]['stock_on_hand']:.2f}")
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    print("=" * 50)
    print("BiLSTM Prediction API Test Client")
    print("=" * 50)

    try:
        # Run all tests
        test_health_check()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_with_real_sequence()

        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the FastAPI server is running on http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")