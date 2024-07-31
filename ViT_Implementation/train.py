import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor, Lambda
from torchvision.datasets import CIFAR10, ImageNet
from torchvision.transforms import Compose, Resize, ToTensor
from datasets import load_dataset
from tqdm import tqdm, trange
import argparse
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sm
import pandas as pd


from model import ViT

# Code is adapted from: https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# Credit goes to Brian Pulfer
 
def main(n_heads, n_blocks, hidden_dim, layer, res, pos, train=True):
    # Loading data
    # transform = ToTensor()
    transform = Compose([
        Resize((64, 64)),
        Lambda(lambda x: x.convert('RGB')),
        ToTensor()
    ])

    def apply_transform(example):
        #print(type(example['image']))
        example['image'] = transform(example['image'])
        #print(type(example['image']))
        return example
    
    dataset = load_dataset("Maysee/tiny-imagenet")
    # print(dataset)

    filtered_train_set = []
    filtered_test_set = []

    for i, label in enumerate(dataset['train']['label']):
        if label < 10:
            filtered_train_set.append(i)

    for i, label in enumerate(dataset['valid']['label']):
        if label < 10:
            filtered_test_set.append(i)

    train_set = dataset['train'].select(filtered_train_set).map(apply_transform)
    test_set = dataset['valid'].select(filtered_test_set).map(apply_transform)
    
    train_set.set_format(type='torch', columns=['image', 'label'])
    test_set.set_format(type='torch', columns=['image', 'label'])
    
    #print(f"size of training set: {len(train_set)}")
    #print(f"size of test set: {len(test_set)}")
    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=128)

    image = train_set[0]['image']  # Get the first image from the dataset (ignoring the label)
    image_dims = image.shape
    #print(image_dims)
    
    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #n_patches = 8, n_blocks=4, hidden_dim = 16, n_heads=4, n_classes=10, layer=True, res=True, pos="trig"):
    model = ViT(
        image_dims,
        n_blocks=n_blocks,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_classes=10,
        res=res,
        layer=layer,
        pos=pos
    ).to(device)

    filename_base = f"{n_heads}_{n_blocks}_{hidden_dim}_{layer}_{res}_{pos}"
    model_name = f"trained_models/{filename_base}.pth"
    cm_path = f"cm/{filename_base}.png"

    LR = 0.01
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    if train:
        N_EPOCHS = 100

        # Training loop
        train_accuracy = 0
        for epoch in trange(N_EPOCHS, desc="Training"):
            train_loss = 0.0
            t_correct, t_total = 0, 0
            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
            ):
                x = batch['image']
                y = batch['label']
                #print("train", y)
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
            train_accuracy = t_correct / t_total * 100
            # print(f"Train accuracy: {train_accuracy:.2f}%")
            # print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
        
        print(f"Trained model with {n_heads} heads, {n_blocks} blocks, {hidden_dim} dims, {layer} norm, {res} res {pos} pos with train accuracy {train_accuracy:.2f}%")
    else:
        model.load_state_dict(torch.load(model_name))

    # Test loop
    test_accuracy = 0
    y_total = []
    yhat_total = []
    yhat_total_no_argmax = []
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x = batch['image']
            y = batch['label']
            y_total.extend(y.tolist())
            #print("test", y)
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            yhat_total.extend([np.argmax(e) for e in y_hat.cpu().numpy()])
            yhat_total_no_argmax.extend([e for e in y_hat.cpu().numpy()])
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        # print(f"Test loss: {test_loss:.2f}")
        test_accuracy = correct / total * 100
        print(f"Test accuracy: {test_accuracy:.2f}%\n")
    # save the model to disk

    # all_labels = np.array(y_total)
    # all_outputs = np.array(yhat_total)
    # y_scores = np.array(yhat_total_no_argmax)
    # print(all_labels[:20])
    # print(all_outputs[:20])

    # print(f"Tested model with {n_heads} heads, {n_blocks} blocks, {hidden_dim} dims, {layer} norm, {res} res with test accuracy {test_accuracy:.2f}%")

    # cm = confusion_matrix(all_labels, all_outputs)
    # print("Confusion Matrix:", cm)

    # classes = range(10)
    # # Build a dataframe
    # df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index = [i for i in classes],
    #                     columns = [i for i in classes])

    # # Plot the heatmap with the custom colormap
    # cmap = plt.cm.get_cmap('Greens', 5)
    # plt.figure(figsize = (12,7))
    # sm.heatmap(df_cm, annot=True, cmap=cmap, annot_kws={"size": 16})
    # plt.title(f'Confusion Matrix {filename_base}', fontsize = 12)
    # plt.xlabel('Predicted Class', fontsize = 18)
    # plt.ylabel('True Class', fontsize = 18)
    # plt.xticks(np.arange(0.5, classes[-1] + 1.5), classes, fontsize = 20)
    # plt.yticks(np.arange(0.5, classes[-1] + 1.5), classes, fontsize = 20)
    # print(cm_path)
    # plt.savefig(cm_path)

    # f1 = f1_score(all_labels, all_outputs, average="macro")
    # print("F1 Score:", f1)

    # precision = precision_score(all_labels, all_outputs, average="macro")
    # print("Precision: ", precision)

    # recall = recall_score(all_labels, all_outputs, average="macro")
    # print("Recall: ", recall)

    # roc_auc = roc_auc_score(all_labels, y_scores, multi_class='ovr')
    # print('ROC AUC Score: ', roc_auc)

    

    if train:
        torch.save(model.state_dict(), model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This takes in the transformer hparams"
    )

    parser.add_argument(
        "--hidden_dim", "-d", type=int, default=16, help="hidden dimension size"
    )
    parser.add_argument(
        "--n_heads", "-n", type=int, default=4, help="number of heads"
    )
    parser.add_argument(
        "--n_patches", "-p", type=int, default=8, help="number of patches"
    )
    parser.add_argument(
        "--n_blocks", "-b", type=int, default=4, help="number of blocks"
    )
    parser.add_argument(
        "--pos", type=str, default="trig", help="type of positional encoding scheme"
    )

    args = parser.parse_args()
#    def main(n_heads,      n_blocks,      hidden_dim, layer, res, pos, train=True):
    main(args.n_heads, args.n_blocks, args.hidden_dim, True,  True, args.pos)