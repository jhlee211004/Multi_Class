# ============================
#  RF: Large-Search Re-Tuning
# ============================
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# 1. Load preprocessed data
# ----------------------------
X_train = np.load("X_train_scaled.npy")
X_test  = np.load("X_test_scaled.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

print("Loaded datasets!")
print("Train:", X_train.shape, "Test:", X_test.shape)

# ----------------------------
# 2. Large Search Space
# ----------------------------
param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [40, 60, 80, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}

print("\n================== Re-Tuning RandomForest (Large Grid) ==================")
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest RF parameters:", grid.best_params_)

# ----------------------------
# 3. Evaluate tuned model
# ----------------------------
best_rf = grid.best_estimator_

y_pred = best_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🔥 RandomForest Re-Tuned Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# ----------------------------
# 4. Save best model
# ----------------------------
with open("best_rf_large.pkl", "wb") as f:
    pickle.dump(best_rf, f)

print("\nModel saved as best_rf_large.pkl")
