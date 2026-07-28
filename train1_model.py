import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("vehicle_resale_dataset.csv")

# Encode categorical columns
categorical_cols = [
    "fuel_type",
    "brand",
    "transmission",
    "color",
    "service_history",
    "insurance_valid"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features
X = df.drop("price_usd", axis=1)

# Target
y = df["price_usd"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)

print("R2 Score :", r2_score(y_test, pred))

# Save model
joblib.dump(model, "model.pkl")

print("model.pkl created successfully!")