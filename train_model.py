import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Show dataset info
print(df.info())

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove extra spaces from string values
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Encode ALL categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Check dataset after encoding
print("\nAfter Encoding:\n")
print(df.head())

# Check Columns 
print(df.columns)
# Features and Target
X = df.drop(["PlacementStatus", "SSC_Marks", "HSC_Marks"], axis=1)
y = df["PlacementStatus"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Save Model
pickle.dump(model, open("placement_model.pkl", "wb"))

print("\nModel saved successfully!")