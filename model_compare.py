import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# =====================
# Load processed dataset
# =====================
X_train = np.load("X_train_scaled.npy")
X_test  = np.load("X_test_scaled.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

print("Loaded processed datasets.")
print("Train:", X_train.shape, "Test:", X_test.shape)

# =====================
# Define 4 models
# =====================
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42
    ),

    "SVM_RBF": SVC(
        kernel='rbf', C=3, gamma='scale'
    ),

    "MLP_NN": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=40,
        random_state=42
    )
}

# =====================
# Train & Evaluate
# =====================
results = {}

for name, model in models.items():
    print(f"\n================= {name} =================")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, pred))

    results[name] = acc

print("\n================= SUMMARY =================")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
