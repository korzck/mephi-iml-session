# Итоговый проект по дисциплине "Внедрение моделей машинного обучения"

**Выполнил студент:** M255589

ML-сервис для прогнозирования дефолта по кредитным картам с контейнеризацией и A/B-тестированием.

---

## Описание проекта

**Цель:** разработать и внедрить в production-like-среду сервис машинного обучения для прогнозирования дефолта по кредитным картам.

**Домен:** финансы / кредитный скоринг

**Датасет:** Default of Credit Card Clients Dataset (UCI ML Repository)

- Клиенты кредитных карт на Тайване
- Таргет: `default.payment.next.month` (дефолт в следующем месяце)
- Признаки: демографические данные, история платежей, суммы счетов

---

## Структура репозитория

```
.
|- app/               # Flask-приложение
|  |- api.py
|- data/              # датасет
|  |- UCI_Credit_Card.csv
|- docker/            # Dockerfile
|  |- Dockerfile
|- models/            # скрипт обучения и .pkl файлы
|  |- train_model.py
|  |- model_v1.pkl
|  |- model_v2.pkl
|- tests/             # тесты
|  |- test_app.py
|- docker-compose.yml
|- requirements.txt
|- README.md
```

---

## Инструкция по запуску

### Способ 1: Docker

```bash
# скачать образ с Docker Hub
docker pull korzck/credit-card-ml:latest

# запустить контейнер
docker run -d -p 5000:5000 --name credit-card-ml korzck/credit-card-ml:latest
```

### Способ 2: Локальная разработка

```bash
# клонировать репозиторий
git clone https://github.com/korzck/mephi-iml-session.git
cd mephi-iml-session

# создать виртуальное окружение
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# установить зависимости
pip3 install -r requirements.txt

# запустить сервер
python3 app/api.py
```

### Способ 3: Docker Compose

```bash
docker compose up --build
```

---

## API Endpoints

### GET /health

Проверка работоспособности сервиса.

**Пример запроса:**

```bash
curl http://localhost:5000/health
```

**Ответ:**

```json
{
  "status": "ok",
  "message": "Service is running",
  "models_loaded": ["v1", "v2"]
}
```

### POST /predict

Прогнозирование дефолта клиента.

**Параметры запроса:**

| Поле       | Тип    | Описание                                 |
| ---------- | ------ | ---------------------------------------- |
| LIMIT_BAL  | int    | кредитный лимит                          |
| SEX        | int    | пол (1 - муж, 2 - жен)                   |
| EDUCATION  | int    | образование (1-4)                        |
| MARRIAGE   | int    | семейное положение (1-3)                 |
| AGE        | int    | возраст                                  |
| PAY_0..6   | int    | статус платежа за последние 6 месяцев    |
| BILL_AMT1..6 | float | сумма счета за последние 6 месяцев     |
| PAY_AMT1..6  | float | сумма платежа за последние 6 месяцев   |
| user_id    | string | (опционально) для A/B-тестирования       |

**Пример запроса:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000,
    "AGE": 30,
    "SEX": 1,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 5000,
    "BILL_AMT2": 4500,
    "BILL_AMT3": 4000,
    "BILL_AMT4": 3500,
    "BILL_AMT5": 3000,
    "BILL_AMT6": 2500,
    "PAY_AMT1": 1000,
    "PAY_AMT2": 900,
    "PAY_AMT3": 800,
    "PAY_AMT4": 700,
    "PAY_AMT5": 600,
    "PAY_AMT6": 500
  }'
```

**Ответ:**

```json
{
  "prediction": 0,
  "probability_default": 0.14,
  "model_version": "v1"
}
```

- `prediction` - 1 если дефолт, 0 если нет
- `probability_default` - вероятность дефолта от 0 до 1
- `model_version` - какая модель обработала запрос (v1 или v2)

---

## Docker

Docker Hub: https://hub.docker.com/r/korzck/credit-card-ml

```bash
docker pull korzck/credit-card-ml:latest
```

Сборка образа локально:

```bash
docker build -f docker/Dockerfile -t credit-card-ml:latest .
```

Запуск контейнера:

```bash
docker run -p 5000:5000 credit-card-ml:latest
```

---

## Модели и метрики

### Model v1 - RandomForestClassifier (n_estimators=100)

Pipeline: StandardScaler + RandomForest, 23 признака

| Метрика   | Значение |
| --------- | -------- |
| Accuracy  | 0.8163   |
| F1-score  | 0.4661   |
| Precision | 0.6405   |
| Recall    | 0.3663   |

### Model v2 - GradientBoostingClassifier (n_estimators=100, lr=0.1)

Pipeline: StandardScaler + GradientBoosting, 23 признака

| Метрика   | Значение |
| --------- | -------- |
| Accuracy  | 0.8200   |
| F1-score  | 0.4627   |
| Precision | 0.6671   |
| Recall    | 0.3542   |

v2 дает Precision выше на 2.6%, то есть меньше ложных отказов хорошим клиентам. Именно это проверяется в A/B-тесте.

---

## Оптимизация модели с ONNX-ML

ONNX (Open Neural Network Exchange) - формат для переносимости и ускорения инференса ML-моделей.

**Как преобразовать модель:**

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# указываем тип и размерность входных данных
initial_type = [("float_input", FloatTensorType([None, 23]))]

# конвертируем sklearn pipeline в ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**Преимущества:** latency снижается в 2-3 раза, модель можно запустить на ONNX Runtime без Python/sklearn.

---

## Production-среда: uWSGI + NGINX

**Почему Flask не подходит для production?**

Flask встроенный сервер:
- однопоточный, один запрос за раз
- не рассчитан на высокую нагрузку
- нет механизмов перезапуска при падении

**Решение - uWSGI + NGINX:**

- **uWSGI** - запускает несколько воркеров (копий приложения), управляет памятью и процессами
- **NGINX** - стоит перед uWSGI, принимает запросы, балансирует нагрузку, защищает от DDoS

Схема:
```
Client -> NGINX (порт 80) -> uWSGI (сокет) -> Flask App
```

---

## Архитектура сервиса

### Монолит vs микросервисы

В данном проекте выбран **монолитный подход**.

**Обоснование:**
- простота деплоя - один Docker-контейнер
- нет сетевых вызовов между сервисами, меньше latency
- для учебного проекта с одной моделью этого достаточно
- легче отлаживать и тестировать

**Когда переходить на микросервисы:**
- разные команды работают над разными частями системы
- компоненты нужно масштабировать независимо (например, инференс отдельно от API)
- нужна независимая выкатка новых версий модели без остановки всего сервиса

### Брокеры сообщений (RabbitMQ)

В сценарии масштабирования RabbitMQ может использоваться для:

1. **Асинхронной обработки** - клиент отправляет запрос в очередь, ML-сервис забирает и обрабатывает, результат возвращается через callback
2. **Batch-предсказаний** - накапливаем запросы в очереди и обрабатываем пачками раз в N минут
3. **Логирования** - все предсказания пишем в очередь логов, отдельный сервис-логгер пишет в Elasticsearch

Пример схемы с брокером:

```
Client -> RabbitMQ (queue) -> ML Service -> Prediction
                                   |
                             Logging Service -> Elasticsearch
```

---

## Логирование и мониторинг

Каждый запрос к `/predict` логируется в JSON-формате в stdout:

```json
{
  "timestamp": "2026-04-16T10:00:00Z",
  "endpoint": "/predict",
  "status": 200,
  "model_version": "v1",
  "prediction": 0,
  "probability_default": 0.14,
  "latency_ms": 13.8,
  "user_id": "user_42"
}
```

Docker Compose настроен на сбор логов через `json-file` драйвер с ротацией (max-size: 10m, max-file: 3).

**В production-среде** логи можно передавать в ELK-стек:
- Filebeat/Fluentd забирает JSON-логи из Docker
- Elasticsearch индексирует и хранит
- Kibana строит дашборды по latency, распределению предсказаний, метрикам A/B-теста

---

## A/B Тестирование

### Практическая демонстрация

Сервис поддерживает A/B-тестирование через поле `user_id` в запросе. Один и тот же клиент всегда попадает в одну группу (детерминированный хеш).

**Запрос к модели v1 (user_1 -> v1):**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_1", "LIMIT_BAL": 50000, "AGE": 30, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 5000, "BILL_AMT2": 4500, "BILL_AMT3": 4000, "BILL_AMT4": 3500, "BILL_AMT5": 3000, "BILL_AMT6": 2500, "PAY_AMT1": 1000, "PAY_AMT2": 900, "PAY_AMT3": 800, "PAY_AMT4": 700, "PAY_AMT5": 600, "PAY_AMT6": 500}'
# ответ: "model_version": "v1"
```

**Запрос к модели v2 (user_2 -> v2):**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_2", "LIMIT_BAL": 50000, "AGE": 30, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 5000, "BILL_AMT2": 4500, "BILL_AMT3": 4000, "BILL_AMT4": 3500, "BILL_AMT5": 3000, "BILL_AMT6": 2500, "PAY_AMT1": 1000, "PAY_AMT2": 900, "PAY_AMT3": 800, "PAY_AMT4": 700, "PAY_AMT5": 600, "PAY_AMT6": 500}'
# ответ: "model_version": "v2"
```

Поле `model_version` в ответе позволяет агрегировать метрики по группам и считать статистику.

### Постановка эксперимента

**Цель:** сравнить текущую модель v1 (RandomForest) с новой v2 (GradientBoosting).

**Гипотеза:** v2 показывает более высокий F1-score при том же или лучшем Precision.

### Разделение трафика

**Сплит:** 50/50

**Метод:** по `user_id` через хеш-функцию - один клиент всегда в одной группе

```python
import hashlib

def get_ab_group(user_id: str) -> str:
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "v2" if h % 2 == 0 else "v1"
```

### Метрики оценки

| Метрика                    | Тип            | Обоснование                                                   |
| -------------------------- | -------------- | ------------------------------------------------------------- |
| F1-score (класс default=1) | Основная       | баланс между Precision и Recall, оба важны для банка          |
| Precision (класс default=1)| Дополнительная | цена ложного срабатывания высока - клиенту откажут в кредите  |

Recall тоже важен, но слишком высокий Recall за счет Precision дает много ложных отказов хорошим клиентам - это прямые убытки.

### Продолжительность теста

**Минимальный срок:** 14 дней

**Целевой объем:** минимум 1000 предсказаний на каждую группу

```
Длительность = 2000 / суточный_трафик
При трафике 200 запросов в день: 2000 / 200 = 10 дней
```

### Статистический анализ

**Метод:** двухвыборочный z-тест для пропорций

```
z = (p_test - p_control) / sqrt(p_pooled * (1 - p_pooled) * (1/n_test + 1/n_control))
```

Доверительный интервал (95%):

```
(p_test - p_control) +/- 1.96 * SE
```

### Критерий успешности

| Исход                       | Решение                                     |
| --------------------------- | ------------------------------------------- |
| p-value < 0.05 и dF1 > 0.02 | выкатываем v2 - статистически лучше         |
| p-value >= 0.05             | ничья, оставляем v1, собираем больше данных |
| p-value < 0.05 и dF1 < 0    | останавливаем тест, v2 хуже v1              |

Дополнительный критерий: если F1 не изменился, но Precision вырос на >5% - тоже переходим на v2 (меньше ложных отказов).

---

## Бизнес-метрики

Помимо технических метрик (F1, Precision), для заказчика важны:

### 1. Снижение ожидаемых финансовых потерь

Банк теряет деньги на каждом пропущенном дефолте. Считаем по выходам модели:

```python
# avg_loss - средний долг одного дефолтного клиента
expected_loss = false_negatives * avg_loss
# чем выше Recall у модели, тем меньше пропущенных дефолтов и меньше потери
improvement = expected_loss_v1 - expected_loss_v2
```

### 2. Доля одобренных заявок при фиксированном уровне риска

Банк хочет одобрить максимум заявок, не превышая допустимый процент дефолтов (например, не более 5%):

```python
# порог подбирается так, чтобы predicted_default_rate <= 0.05
approval_rate = sum(prediction == 0) / total_requests
# модель лучше, если при том же пороге у нее выше approval_rate
```

Каждый дополнительно одобренный клиент - это доход от процентов по кредиту.

---

## MLOps: DVC и MLflow

### DVC (Data Version Control)

DVC решает проблему воспроизводимости в части данных и артефактов:

- **Версионирование данных** - `data/UCI_Credit_Card.csv` хранится в S3/GCS, в Git только `.dvc`-файл-указатель
- **Версионирование моделей** - `models/model_v1.pkl` (53 MB) аналогично выносится из репозитория
- **Пайплайны** - `dvc.yaml` описывает шаги `data -> train -> evaluate`, воспроизвести можно командой `dvc repro`

В нашем проекте DVC позволил бы хранить большие файлы за пределами Git и гарантировать, что любой получит тот же `model_v1.pkl` из того же датасета.

### MLflow

MLflow решает проблему отслеживания экспериментов:

- **Tracking** - каждый запуск `train_model.py` логирует гиперпараметры, метрики и сохраняет `.pkl`
- **Model Registry** - модели имеют статусы `Staging` / `Production`, из `Production` берется модель для деплоя
- **Сравнение** - UI MLflow позволяет сравнить v1 и v2 по всем метрикам в одной таблице

```python
import mlflow

with mlflow.start_run(run_name="model_v2_GradientBoosting"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("f1_score", 0.4627)
    mlflow.log_metric("precision", 0.6671)
    mlflow.sklearn.log_model(model_v2, "model_v2")
```
