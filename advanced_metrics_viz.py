import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# Load preprocessed datasets
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Example predictions (Replace these with actual model predictions)
predictions = {
    'Logistic Regression': np.load('logistic_regression_preds.npy'),
    'Random Forest': np.load('random_forest_preds.npy'),
    'XGBoost': np.load('xgboost_preds.npy'),
}

# Store metrics for visual comparison
metrics = {'Model': [], 'Accuracy': [], 'F1 Score': [], 'ROC AUC': []}

# Function to calculate Specificity and Sensitivity
def calculate_specificity_sensitivity(cm):
    TN, FP, FN, TP = cm.ravel()
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    return specificity, sensitivity

# Visualize Precision-Recall Curve
def plot_precision_recall_curve(y_test, y_pred_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(f'{model_name}_precision_recall_curve.png')
    plt.close()
    print(f"Precision-Recall Curve saved for {model_name}.")

# Loop through each model to calculate metrics and plot results
for model_name, y_pred in predictions.items():
    # Classification report and confusion matrix
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Calculate Specificity and Sensitivity
    specificity, sensitivity = calculate_specificity_sensitivity(cm)
    print(f"{model_name} Specificity: {specificity:.2f}, Sensitivity: {sensitivity:.2f}")

    # Store metrics for comparison
    metrics['Model'].append(model_name)
    metrics['Accuracy'].append(report['accuracy'])
    metrics['F1 Score'].append(report['1']['f1-score'])
    metrics['ROC AUC'].append(roc_auc_score(y_test, y_pred))

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
    print(f"Confusion Matrix saved for {model_name}.")

    # Plot Precision-Recall Curve (assuming probabilities are available)
    y_pred_proba = np.load(f'{model_name.lower().replace(" ", "_")}_probas.npy')  # Replace with actual probabilities
    plot_precision_recall_curve(y_test, y_pred_proba, model_name)

# Convert metrics to DataFrame for visualization
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('model_metrics_comparison.csv', index=False)

# Plot bar charts for metrics comparison
plt.figure(figsize=(10, 6))
x = np.arange(len(metrics['Model']))
width = 0.2

plt.bar(x - width, metrics['Accuracy'], width, label='Accuracy')
plt.bar(x, metrics['F1 Score'], width, label='F1 Score')
plt.bar(x + width, metrics['ROC AUC'], width, label='ROC AUC')

plt.xticks(x, metrics['Model'])
plt.ylabel('Scores')
plt.title('Model Metrics Comparison')
plt.legend()
plt.savefig('model_comparison_bar_plot.png')
plt.close()

print("Model metrics comparison and visualizations saved.")
