import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import joblib
import os

# загружаем датасет
df = pd.read_csv("data/UCI_Credit_Card.csv")
print(df.head())

# готовим признаки и целевую переменную
X = df.drop(columns=["ID", "default.payment.next.month"])
y = df["default.payment.next.month"]

# делим на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os.makedirs("models", exist_ok=True)

# --- Модель v1: RandomForest ---
# пайплайн удобен тем, что сохраняет всю цепочку преобразований вместе с моделью
model_v1 = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

print("Обучение модели v1 (RandomForest)...")
model_v1.fit(X_train, y_train)
joblib.dump(model_v1, "models/model_v1.pkl")
print("Модель v1 сохранена в models/model_v1.pkl")

y_pred_v1 = model_v1.predict(X_test)
print("\n=== Метрики модели v1 (RandomForest) ===")
print(f"Accuracy:  {model_v1.score(X_test, y_test):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_v1):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_v1):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_v1):.4f}")
print(classification_report(y_test, y_pred_v1, target_names=["no_default", "default"]))

# --- Модель v2: GradientBoosting (для A/B теста) ---
model_v2 = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
])

print("Обучение модели v2 (GradientBoosting)...")
model_v2.fit(X_train, y_train)
joblib.dump(model_v2, "models/model_v2.pkl")
print("Модель v2 сохранена в models/model_v2.pkl")

y_pred_v2 = model_v2.predict(X_test)
print("\n=== Метрики модели v2 (GradientBoosting) ===")
print(f"Accuracy:  {model_v2.score(X_test, y_test):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_v2):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_v2):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_v2):.4f}")
print(classification_report(y_test, y_pred_v2, target_names=["no_default", "default"]))
