"""predict.

Usage: predict <model> <filepath> [--map <class_to_value>] [--cpu|--gpu] [options]

Options:
    --gpu                   Train the model on the GPU.
    --cpu                   Train the model on the CPU.
    --top-k <topk>          Top k probabilities. [default: 5]
    -d --debug              Enable debugging mode.
"""

from docopt import docopt
import torch
from model import build_model
from PIL import Image
import numpy as np
import json


def process_image(path):
    image = Image.open(path)
    # Process a PIL image for use in a PyTorch model
    image = image.resize([256, 256])
    offset = (256 - 224) / 2
    image = image.crop((offset, offset, 256 - offset, 256 - offset))

    np_image = np.array(image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_image = (image - mean) / std

    return norm_image.transpose(2, 0, 1)


def predict(image_path, model, topk=5):
    image = torch.from_numpy(process_image(image_path))
    image = image.reshape(1, 3, 224, 224).float()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    model.eval()
    ps = torch.exp(model(image))
    top_p, top_class = ps.topk(topk, dim=1)
    return top_p[0].tolist(), [idx_to_class[c.item()] for c in top_class[0]]


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    arch = checkpoint["arch"]
    hl = checkpoint["classifier"]["hidden_layers"]
    del hl[0]
    model = build_model(arch, hl)

    model.load_state_dict(checkpoint['state_dict'])
    model.classifier.load_state_dict(checkpoint['classifier']['state_dict'])

    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def main():
    arguments = docopt(__doc__)
    if arguments["--gpu"] and not torch.cuda.is_available():
        print("GPU not supported. Falling back to CPU")
        device = "cpu"
    elif arguments["--gpu"] and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    cat_to_name = None

    if arguments["--map"]:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)

    model = load_checkpoint(arguments["<model>"])
    model.to(device)
    probabilities, classes = predict(arguments["<filepath>"], model, int(arguments["--top-k"]))

    label = cat_to_name[classes[0]] if cat_to_name is not None else classes[0]
    print(f"The flower is {label} with a probability of {probabilities[0]}")

    print("The following are the topk classifications.")
    for probability, class_ in zip(probabilities[1:], classes[1:]):
        label = cat_to_name[class_] if cat_to_name is not None else class_
        print(f"Classified as {label} with a probability of {probability}")


if __name__ == "__main__":
    main()
