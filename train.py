import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("nsl_kdd_dataset.csv")

# Use limited features (simple & safe)
features = ["duration", "src_bytes", "dst_bytes"]
target = "label"

X = df[features]
y = df[target]

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save everything
pickle.dump(
    (model, scaler, label_encoder),
    open("model.pkl", "wb")
)

print("Model trained and saved successfully")
