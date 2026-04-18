from minisom import MiniSom
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

# Data
data = np.array([[1, 2], [2, 3], [3, 4], [8, 9], [9, 10]])

# Create and train SOM
som = MiniSom(3, 3, 2)
som.train(data, 100)

# Simple loop
for i in range(len(data)):
    print("Data point", data[i], "mapped to neuron", som.winner(data[i]))

# Ground truth labels (0 for cluster 1, 1 for cluster 2)
y_true = np.array([0, 0, 0, 1, 1])

# Get SOM predictions by quantizing winner positions to binary classes
y_pred = []
for x in data:
    winner = som.winner(x)
    # Quantize neuron position to class (top/left = 0, bottom/right = 1)
    predicted_class = 0 if (winner[0] + winner[1]) < 2 else 1
    y_pred.append(predicted_class)
y_pred = np.array(y_pred)

# Calculate classification metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\nClassification Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
