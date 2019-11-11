"""train.

Usage: train <dataset> [--cpu|--gpu] [--hidden-units hu...]  [options]

Options:
    --gpu                   Train the model on the GPU.
    --cpu                   Train the model on the CPU.
    --learning-rate lr      Set the learning rate. [default: 0.001]
    --epochs epoch          Set the number of epochs. [default: 5]
    --hidden-units hu       Set the number of hidden units.
    --arch arch             The architecture for the NN. [default: vgg16]
    -d --debug              Enable debugging mode.
"""

from docopt import docopt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from model import build_model


def load_dataset(path):
    image_size = 224
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(image_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    return {
        "training": train_loader,
        "testing": test_loader,
        "validation": valid_loader
    }


def train(data, model, lr=0.001, epochs=5, device="cpu"):
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    model.to(device)
    for epoch in range(epochs):
        loss_accum = 0

        for image, labels in data["training"]:
            image, labels = image.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_accum += loss.item()
        else:
            validation_loss_accum = 0
            validation_accuracy_accum = 0

            model.eval()
            with torch.no_grad():
                for image, labels in data["validation"]:
                    image, labels = image.to(device), labels.to(device)

                    log_ps = model(image)
                    validation_loss_accum = criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    validation_accuracy_accum += torch.mean(equals.type(torch.FloatTensor)).item()
            model.train()

            print(f"Training loss: {loss_accum / len(data['training'])}")
            print(f"Validation loss: {validation_loss_accum / len(data['validation'])}")
            print(f"Validation accuracy: {validation_accuracy_accum / len(data['validation']) * 100}%")


def is_valid_architecture(arch):
    class_ = getattr(models, arch)
    return class_ is not None


def save_model(filename, dataset, model, arch, hidden_layers):
    model.class_to_idx = dataset["training"].dataset.class_to_idx

    checkpoint = {'class_to_idx': model.class_to_idx,
                  'arch': arch,
                  'classifier': {
                      "state_dict": model.classifier.state_dict(),
                      "hidden_layers": hidden_layers,
                  },
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, filename)


def main():
    arguments = docopt(__doc__)

    data = load_dataset(arguments["<dataset>"])

    output_dimension = len(data["training"].dataset.classes)
    hl = arguments["--hidden-units"] if arguments["--hidden-units"] else [2000, 1000]
    hl = [int(h) for h in hl]
    hl.append(output_dimension)

    if arguments["--gpu"] and not torch.cuda.is_available():
        print("GPU not supported. Falling back to CPU")
        device = "cpu"
    elif arguments["--gpu"] and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if is_valid_architecture(arguments["--arch"]):
        model = build_model(arguments["--arch"], hl)
        train(data, model, device=device, lr=float(arguments["--learning-rate"]), epochs=int(arguments["--epochs"]))
        save_model("checkpoint.pth", data, model, arguments["--arch"], hl)
    else:
        print(f'Architecture ${arguments["--arch"]} not available.')


if __name__ == "__main__":
    main()
