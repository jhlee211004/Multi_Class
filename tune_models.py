import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# Load processed data
# =========================================================
X_train = np.load("X_train_scaled.npy")
X_test  = np.load("X_test_scaled.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

print("Loaded datasets!")
print("Train:", X_train.shape, "Test:", X_test.shape)


# =========================================================
# 1) RandomForest Hyperparameter Search
# =========================================================
print("\n================== Tuning RandomForest ==================")

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

rf_param_dist = {
    "n_estimators": randint(200, 700),
    "max_depth": [None, 20, 40, 60],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5)
}

rf_search = RandomizedSearchCV(
    rf, 
    rf_param_dist,
    n_iter=20,
    scoring="accuracy",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

rf_search.fit(X_train, y_train)

print("\nBest RF parameters:", rf_search.best_params_)

best_rf = rf_search.best_estimator_
rf_pred = best_rf.predict(X_test)

print("\nRandomForest Tuned Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


# =========================================================
# 2) GradientBoosting Hyperparameter Search
# =========================================================
print("\n================== Tuning GradientBoosting ==================")

gb = GradientBoostingClassifier(random_state=42)

gb_param_dist = {
    "n_estimators": randint(100, 500),
    "learning_rate": uniform(0.01, 0.2),
    "max_depth": randint(2, 6),
    "subsample": uniform(0.7, 0.3)
}

gb_search = RandomizedSearchCV(
    gb,
    gb_param_dist,
    n_iter=20,
    scoring="accuracy",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

gb_search.fit(X_train, y_train)

print("\nBest GB parameters:", gb_search.best_params_)

best_gb = gb_search.best_estimator_
gb_pred = best_gb.predict(X_test)

print("\nGradientBoosting Tuned Accuracy:", accuracy_score(y_test, gb_pred))
print(classification_report(y_test, gb_pred))


# =========================================================
# Final Comparison Summary
# =========================================================
print("\n================== FINAL SUMMARY ==================")
print("RandomForest (tuned):", accuracy_score(y_test, rf_pred))
print("GradientBoosting (tuned):", accuracy_score(y_test, gb_pred))
