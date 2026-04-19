from algorithms.neural_network_core import NNClassifier
from NN1validate import validate_model

"""
Initializes the NNClassifier, executes the training process,
and validates the resulting model against the test dataset.
"""
instance = NNClassifier()                                              # create new model instance

instance.train_model()                                                 # trains the model
accuracy, error = validate_model(instance)                             # validates the model



"""
RESULTS
"""
print("\nHidden Layers | Accuracy | Error")
print("----------------------------------")
print(f"{'1 layers':<13} | {accuracy:.4f} | {error:.4f}")