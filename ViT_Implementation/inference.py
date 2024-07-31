import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor, Lambda
from torchvision.datasets import CIFAR10, ImageNet
from torchvision.transforms import Compose, Resize, ToTensor
from datasets import load_dataset
import argparse

import cv2
from PIL import Image

import time


from model import ViT

# Code is adapted from: https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# Credit goes to Brian Pulfer
 
def main(n_heads, n_blocks, hidden_dim, layer, res, pos, train=True):
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)
    cap.set(cv2.CAP_PROP_FPS, 36)
    print('here')
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

    filtered_test_set = []

    for i, label in enumerate(dataset['valid']['label']):
        if label < 10:
            filtered_test_set.append(i)

    train_set = dataset['valid'].select(filtered_test_set).map(apply_transform)
    test_set = dataset['train'].select(filtered_test_set).map(apply_transform)
    
    test_set.set_format(type='torch', columns=['image', 'label'])
    
    
    #print(f"size of train set: {len(train_set)}")
    #print(f"size of test set: {len(test_set)}")
    test_loader = DataLoader(test_set, shuffle=True, batch_size=128)

    #image = test_set[0]['image']  # Get the first image from the dataset (ignoring the label)
    image_dims = image.shape
    #print(f"test set type: {type(test_set)}")
    print(image_dims)

    # # Display image
    # # Convert to numpy array
    # img = image.permute(1, 2, 0).numpy()

    # # Normalize to [0, 1] range for displaying
    # img = (img - img.min()) / (img.max() - img.min())

    # # Display the image
    # plt.imshow(img)
    # plt.axis('off')  # Turn off axis labels
    # plt.show()
    
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

    model.load_state_dict(torch.load(model_name))
    
    started = time.time()
    last_logged = time.time()
    frame_count = 0

    class_labels = ["Fish", "Salamander", "Tailed Frog", "Bullfrog", "American Alligator", "Boa Constrictor", "Trilobite", "Scorpion", "Black Widow", "Tarantula"]
    with torch.no_grad():
        while True:

    
            ret, image = cap.read()
            image = image[:,:,[2,1,]]

            #input tensor 
            input_tensor = transform(image)

            y = model(input_tensor.unsqueeze(0))
            max_class_index = torch.argmax(y)
            predicted_class_name = class_labels[max_class_index]
            print(predicted_class_name)


             # log model performance
            frame_count += 1
            now = time.time()
            if now - last_logged > 1:
                print(f"{frame_count / (now-last_logged)} fps")
                last_logged = now
                frame_count = 0

    
     # images size = [N, C, H, W]
     # image: torch.Size([3, 64, 64])
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This takes in the transformer hparams"
    )

    parser.add_argument(
        "--hidden_dim", "-d", type=int, default=32, help="hidden dimension size"
    )
    parser.add_argument(
        "--n_heads", "-n", type=int, default=8, help="number of heads"
    )
    parser.add_argument(
        "--n_patches", "-p", type=int, default=8, help="number of patches"
    )
    parser.add_argument(
        "--n_blocks", "-b", type=int, default=1, help="number of blocks"
    )
    parser.add_argument(
        "--pos", type=str, default="learned", help="type of positional encoding scheme"
    )

    args = parser.parse_args()
#    def main(n_heads,      n_blocks,      hidden_dim, layer, res, pos, train=True):
    main(args.n_heads, args.n_blocks, args.hidden_dim, True,  True, args.pos)
