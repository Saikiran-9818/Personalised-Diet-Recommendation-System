import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("🚀 Starting model training process with the new synthetic dataset...")

# --- 1. Configuration & File Paths ---
# Correct path as per your project structure
DATASET_FILE = os.path.join("datasets", "Raw_dataset", "synthetic_health_data.csv")

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"❌ Critical Error: Dataset not found at '{DATASET_FILE}'. Please check the path.")

# --- 2. Load and Prepare Data ---
print(f"🔄 Loading dataset from '{DATASET_FILE}'...")
df = pd.read_csv(DATASET_FILE)

# --- 3. Feature Engineering & Selection ---
# Define the target variable we want to predict
TARGET_COLUMN = "Diet_Recommendation"

# Define the exact feature columns that will be used for training.
# These match the compulsory inputs from your web form.
FEATURE_COLUMNS = [
    'Age', 'Gender', 'Weight_kg', 'Height_cm', 'BMI', 
    'Physical_Activity_Level', 'Occupation', 'Sleep_Duration', 
    'Stress_Level', 'Alcohol_Consumption'
]

# Separate features (X) and target (y)
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print("✅ Dataset loaded successfully.")
print(f"🎯 Target: '{TARGET_COLUMN}'")
print(f"💪 Features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")

# --- 4. Preprocessing ---
print("🔄 Preprocessing data: Encoding categorical features...")

label_encoders = {}
# Create a copy to avoid SettingWithCopyWarning
X = X.copy()

# Identify categorical columns to be encoded
categorical_cols = X.select_dtypes(include=['object']).columns

for col in categorical_cols:
    # Ensure all values are strings to handle mixed types like 'None'
    X[col] = X[col].astype(str)
    
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"   - Encoded '{col}'")

print("✅ Categorical features encoded.")

# --- 5. Train-Test Split ---
print("🔪 Splitting data into training (80%) and testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("✅ Data split successfully.")

# --- 6. Feature Scaling ---
print("⚖️ Scaling numerical features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("✅ Features scaled.")

# --- 7. Model Training ---
print("🧠 Training the RandomForestClassifier model...")
# Using n_jobs=-1 will use all available CPU cores for faster training
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# --- 8. Evaluation ---
print("📊 Evaluating model performance on the test set...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# Because the new dataset is so logical and clean, expect a very high accuracy!
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# --- 9. Save All Necessary Artifacts ---
print("💾 Saving model and all necessary artifacts for the Flask app...")
joblib.dump(model, "diet_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
# Save the exact list of features, which also serves as the column order
joblib.dump(FEATURE_COLUMNS, "column_order.pkl")

print("\n🎉 Success! All files have been saved successfully:")
print("- diet_model.pkl (The new, smarter model)")
print("- scaler.pkl (For scaling user input)")
print("- label_encoders.pkl (For converting user's text input to numbers)")
print("- column_order.pkl (To ensure data is in the correct order for prediction)")

