from algorithms.neural_network_core import NNClassifier

train_path = "data/at_risk_students_training.csv"
test_path = "data/at_risk_students_test.csv"

configs = [
    (5,),
    (10,),
    (20,),
    (32,),
    (16, 8),
    (32, 16),
    (50, 20),
    (20, 20, 10),
    (32, 16, 8),
    (64, 32)
]

print("\nHidden Layers | Accuracy | Error")
print("----------------------------------")

for cfg in configs:

    model = NNClassifier()
    model.model.hidden_layer_sizes = cfg   # override architecture

    model.train_model()
    model.test_model()

    # NOTE: your class currently prints accuracy,
    # so we just observe output