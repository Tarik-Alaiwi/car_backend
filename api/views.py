import pickle
import os
import gdown
import numpy as np
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv

load_dotenv() 

MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_FILE_ID = os.getenv("MODEL_FILE_ID")
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

ENCODER_PATH = os.getenv("ENCODER_PATH")
ENCODER_FILE_ID = os.getenv("ENCODER_FILE_ID")
ENCODER_URL = f"https://drive.google.com/uc?id={ENCODER_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(ENCODER_PATH):
    gdown.download(ENCODER_URL, ENCODER_PATH, quiet=False)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

@csrf_exempt
def predict(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            levy = float(data.get("levy", 0))
            prod_year = int(data.get("prod_year", 0))
            car_age = 2025 - prod_year
            mileage = float(data.get("mileage", 0))
            engine_volume = float(data.get("engine_volume", 0))
            cylinders = float(data.get("cylinders", 0))
            model_name = data.get("model", "").lower().strip()
            try:
                model_label = le.transform([model_name])[0]
            except ValueError:
                return JsonResponse({"error": f"Unknown model: {model_name}"}, status=400)
            X = np.array([[levy, car_age, engine_volume, mileage, cylinders, model_label]])
            prediction = model.predict(X)[0]
            return JsonResponse({"predicted_price": float(prediction)})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "Only POST allowed"}, status=405)