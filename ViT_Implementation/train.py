import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from tqdm import tqdm, trange
import argparse


from ViT_Implementation.model import ViT

# Code is adapted from: https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# Credit goes to Brian Pulfer


def main():
    parser = argparse.ArgumentParser(
        description="This takes in the transformer hparams"
    )

    parser.add_argument(
        "--hidden_dim", "-d", type=int, required=True, help="hidden dimension size"
    )
    parser.add_argument(
        "--n_heads", "-n", type=int, required=True, help="number of heads"
    )
    parser.add_argument(
        "--n_patches", "-p", type=int, required=True, help="number of patches"
    )
    parser.add_argument(
        "--n_blocks", "-b", type=int, required=True, help="number of patches"
    )

    args = parser.parse_args()
    # n_patches=8, n_blocks=2, hidden_dim=8, n_heads=2, n_classes=10)

    # Loading data
    transform = ToTensor()

    train_set = CIFAR10(
        root="./../datasets", train=True, download=True, transform=transform
    )
    test_set = CIFAR10(
        root="./../datasets", train=False, download=True, transform=transform
    )

    print(f"size of training set: {len(train_set)}")
    print(f"size of test set: {len(test_set)}")
    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    image, _ = train_set[0]  # Get the first image from the dataset (ignoring the label)
    image_dims = image.shape

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )

    # load the transformer model and move it to the GPU
    model = ViT(
        image_dims,
        n_patches=args.n_patches,
        n_blocks=args.n_blocks,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_classes=10,
    ).to(device)


    # for name, module in model.named_modules():
    #     print(name, module)
        

    # CIFAR Call
    # model = ViT(image_dims, n_patches=8, n_blocks=2, hidden_dim=8, n_heads=2, n_classes=10).to(device)
    N_EPOCHS = 1
    LR = 0.01

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        t_correct, t_total = 0, 0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.detach().cpu().item() / len(train_loader)
            t_correct += (
                torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            )
            t_total += len(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Train accuracy: {t_correct / t_total * 100:.2f}%")

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
    # save the model to disk
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
