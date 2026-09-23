import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, silhouette_score

DATA = "data/student_performance.csv"
FEATURES = ["Attendance","Study_Hours","Internal_Mark","Assignment","Previous_Mark"]

df = pd.read_csv(DATA)
X = df[FEATURES]

# K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("Silhouette score:", silhouette_score(X_scaled, df["Cluster"]))

# Random Forest
y = df["Performance"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
