from algorithms.knn_core import classification_knn
from collections import Counter

classified_data, test_labels = classification_knn(k_value=3)

# for i, pred in enumerate(classified_data):
#     print(f"Sample {i:<3} → Predicted: {pred} | Actual: {test_labels[i]}")

accuracy = sum(p == t for p, t in zip(classified_data, test_labels)) / len(test_labels)
print(accuracy)

print(Counter(test_labels))