import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import joblib
import json

# =======================
# 1. Load and Clean Dataset
# =======================
df = pd.read_csv(r"heart_disease_uci.csv")

print("Missing values per column:\n", df.isnull().sum())

# Drop irrelevant columns
df.drop(['id', 'dataset'], axis=1, inplace=True)

# Create binary target if needed
if 'target' not in df.columns and 'num' in df.columns:
    df['target'] = (df['num'] > 0).astype(int)
    df.drop('num', axis=1, inplace=True)

# =======================
# 2. Fill Missing Values
# =======================
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

categorical_cols = df.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

assert df.isnull().sum().sum() == 0, "Still has missing values!"

# =======================
# 3. Prepare Features and Labels
# =======================
X = df.drop('target', axis=1)
y = df['target']

# One-hot encode needed columns
X = pd.get_dummies(X, drop_first=True)

# Save feature order for app
feature_order = X.columns.tolist()

# =======================
# 4. Build Preprocessing Pipeline
# =======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the pipeline
joblib.dump(scaler, "pipeline.pkl")

# Save feature order
joblib.dump(feature_order, "feature_order.pkl")

# =======================
# 5. Train-Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# =======================
# 6. Build ANN Model
# =======================
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# =======================
# 7. Train the Model
# =======================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# =======================
# 8. Evaluation
# =======================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc:.4f}")

y_pred = (model.predict(X_test) > 0.5).astype("int32")
y_proba = model.predict(X_test).ravel()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# =======================
# 9. Save Model and History
# =======================
model.save("heart_disease_model.h5")

with open("training_history.json", "w") as f:
    json.dump(history.history, f)

print("\n✅ Model, scaler, and feature order saved successfully!")
