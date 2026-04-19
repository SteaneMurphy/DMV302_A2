from algorithms.neural_network_core import NNClassifier
from NN1validate import validate_model

configs = [                                                            # set configurations of hidden layers and neurons
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

for i in configs:

    instance = NNClassifier()                                         # create instance of model
    instance.model.hidden_layer_sizes = i                             # override default configuration

    instance.train_model()                                            # train model
    accuracy, error = validate_model(instance)                        # validate model

    print(f"{str(i):<14} | {accuracy:.4f}   | {error:.4f}")