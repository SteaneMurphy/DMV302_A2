from algorithms.knn_core import classification_knn, test_evaluation
from collections import Counter

"""
KNN classification experiment.
Runs a KNN-style classifier on the test dataset and checks how accurate it is

Note:
This is not standard weighted KNN (inverse distance weighting).
Instead, it uses a distance-based normalisation where each training sample
contributes based on its distance relative to the total distance of all samples
"""
classified_data, test_labels = classification_knn()
accuracy = test_evaluation(classified_data, test_labels)



"""
RESULTS
"""
print("\n           KNN Evaluation")
print("-------------------------------------")
print(f"Accuracy      : {accuracy:.2f} ({accuracy * 100:.2f}%)")

"""
DATASET DISTRIBUTION
Shows how many samples belong to each class in the test set
"""
label_counts = Counter(test_labels)

print("\nTest label distribution:")
for label, count in sorted(label_counts.items()):
    print(f"Class {label:<2}: {count}")