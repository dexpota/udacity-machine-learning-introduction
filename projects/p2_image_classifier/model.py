from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


def build_model(architecture, hidden_layers):
    class_ = getattr(models, architecture)
    assert class_ is not None

    model = class_(pretrained=True)

    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False

    classifier_input_dimension = list(model.classifier.children())[0].in_features
    classifier_output_dimension = hidden_layers[-1]

    hidden_layers.insert(0, classifier_input_dimension)
    # Add a variable number of more hidden layers
    layer_sizes = zip(hidden_layers[:-2], hidden_layers[1:-1])

    layers = []
    for ii, (h1, h2) in enumerate(layer_sizes):
        layers.append((f'fc{ii}', nn.Linear(h1, h2)))
        layers.append((f'relu{ii}', nn.ReLU()))

    layers.append((f'fc', nn.Linear(hidden_layers[-2], classifier_output_dimension)))
    layers.append((f'logps', nn.LogSoftmax(dim=1)))

    classifier = nn.Sequential(OrderedDict(layers))
    model.classifier = classifier

    return model
