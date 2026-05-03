from flask import Flask, request, jsonify
import joblib
import pandas as pd
import hashlib
import logging
import json
import time
import os

app = Flask(__name__)

# настраиваем логирование в JSON формате - удобно для сбора в ELK
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# список всех признаков модели (порядок важен)
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# словарь с загруженными моделями
models = {}


def load_models():
    # ищем модели относительно корня проекта
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for ver in ("v1", "v2"):
        path = os.path.join(base, "models", f"model_{ver}.pkl")
        if os.path.exists(path):
            models[ver] = joblib.load(path)
            print(f"Модель {ver} загружена из {path}")
        else:
            print(f"Файл модели не найден: {path}")


def get_ab_group(user_id: str) -> str:
    # используем md5 чтобы распределение было стабильным между перезапусками
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "v2" if h % 2 == 0 else "v1"


@app.route("/health", methods=["GET"])
def health_check():
    # простая проверка - живой ли сервис и какие модели загружены
    return jsonify({
        "status": "ok",
        "message": "Service is running",
        "models_loaded": list(models.keys()),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # проверяем что все нужные поля переданы
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({
            "error": f"Missing features: {missing}",
            "required_features": FEATURES,
        }), 400

    # A/B тестирование: если передан user_id - выбираем группу через хеш,
    # иначе по умолчанию отдаем v1
    user_id = data.get("user_id")
    if user_id is not None:
        version = get_ab_group(str(user_id))
    else:
        version = "v1"

    model = models.get(version) or models.get("v1")
    if model is None:
        return jsonify({"error": "Model not available"}), 503

    try:
        input_df = pd.DataFrame([{f: data[f] for f in FEATURES}])
        probability = float(model.predict_proba(input_df)[0][1])
        prediction = int(model.predict(input_df)[0])
    except Exception as e:
        logger.error(json.dumps({"event": "prediction_error", "error": str(e)}))
        return jsonify({"error": str(e)}), 400

    latency_ms = round((time.time() - start) * 1000, 2)

    # логируем каждый запрос в JSON для последующего анализа
    logger.info(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": "/predict",
        "status": 200,
        "model_version": version,
        "prediction": prediction,
        "probability_default": round(probability, 4),
        "latency_ms": latency_ms,
        "user_id": user_id,
    }))

    return jsonify({
        "prediction": prediction,
        "probability_default": round(probability, 4),
        "model_version": version,
    }), 200


if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 5000))
    print(f"Сервис запущен на http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
