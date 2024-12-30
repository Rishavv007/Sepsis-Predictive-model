import numpy as np
from collections import Counter

# Load datasets
y_train = np.load('y.npy')  # Adjust based on your saved file

# Check distribution
class_distribution = Counter(y_train)
print(f"Class distribution in y_train: {class_distribution}")
