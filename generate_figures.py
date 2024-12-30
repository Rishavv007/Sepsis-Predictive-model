import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Example Data
accuracies = [0.91, 0.87, 0.88]  # Replace with actual model accuracies
models = ['LSTM', 'Logistic Regression', 'XGBoost']

# Bar Chart for Accuracy
plt.bar(models, accuracies)
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.savefig('model_comparison.png')

print("Figures saved!")
