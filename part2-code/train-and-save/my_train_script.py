import argparse
import logging
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, input_size, hidden_units, num_classes):
        super().__init__()

        # Initialize MLP layers
        all_layers = []
        for hidden_unit in hidden_units:
            layer = torch.nn.Linear(input_size, hidden_unit, bias=False)
            all_layers.append(layer)
            all_layers.append(torch.nn.ReLU())
            input_size = hidden_unit

        output_layer = torch.nn.Linear(
            in_features=hidden_units[-1], out_features=num_classes
        )

        all_layers.append(output_layer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # to make it work for image inputs
        x = self.layers(x)
        return x


def get_dataloaders(batch_size):
    train_dataset = datasets.MNIST(
        root="data", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = datasets.MNIST(
        root="data", train=False, transform=transforms.ToTensor()
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=1,
        drop_last=True,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, num_workers=1, shuffle=False
    )
    return train_loader, test_loader


def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 28 * 28).to(device)
            targets = targets.to(device)
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float() / num_examples


def train_model(num_epochs, model, train_loader):

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.view(-1, 28 * 28).to(device)
            targets = targets.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            if not batch_idx % 50:
                s = [
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d} | ",
                    f"Batch: {batch_idx:03d}/{len(train_loader):03d} | "
                    f"Loss: {loss:.4f}",
                ]

                logging.info("".join(s))

    train_time = (time.time() - start_time) / 60
    logging.info(f"Total Training Time: {train_time:.2f} min")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--model_out", type=str, default="my_trained_model.pt")
    parser.add_argument("--log_out", type=str, default="log.txt")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(args.log_out), logging.StreamHandler()],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=256)

    model = MultilayerPerceptron(input_size=28 * 28, hidden_units=[50], num_classes=10)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    train_model(args.num_epochs, model, train_loader)

    train_acc = compute_accuracy(model, train_loader)
    test_acc = compute_accuracy(model, test_loader)

    logging.info(f"Training accuracy: {train_acc*100:.2f}%")
    logging.info(f"Test accuracy: {test_acc*100:.2f}%")

    torch.save(model.state_dict(), args.model_out)
