import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
data_path = 'chartevents.csv'  # Update this to your dataset path
chartevents = pd.read_csv(data_path)

# Add a 'sepsis_label' column if not present
if 'sepsis_label' not in chartevents.columns:
    chartevents['sepsis_label'] = 0  # Replace with your own labeling logic

# Preprocessing and feature selection
data = chartevents[['valuenum', 'sepsis_label']].copy()
data.dropna(inplace=True)

scaler = StandardScaler()
data['valuenum'] = scaler.fit_transform(data[['valuenum']])

# Separate features and labels
X = data[['valuenum']].values
y = data['sepsis_label'].values

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Save datasets
np.save('X.npy', X_balanced)
np.save('y.npy', y_balanced)

print("Datasets saved successfully!")
