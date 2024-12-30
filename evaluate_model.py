import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score

# Load datasets and model
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
model = load_model('lstm_sepsis_model.keras')

# Predict and evaluate
y_pred = model.predict(X_test).round()
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

