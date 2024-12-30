import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load the preprocessed dataset
X = np.load("X.npy")
y = np.load("y.npy")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Cross-validation scores for {name}: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean()}")

    # Fit the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Results for {name}:")
    print(report)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {auc}")
    print(f"Confusion Matrix:\n{cm}")

    # Save results
    results[name] = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "ROC AUC Score": auc,
        "Classification Report": report,
        "Confusion Matrix": cm,
    }

    # Save confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"{name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()

# Save results as CSV
results_df = pd.DataFrame(results).T
results_df.to_csv("model_comparison_results.csv", index=True)
print("Results saved to 'model_comparison_results.csv'.")
