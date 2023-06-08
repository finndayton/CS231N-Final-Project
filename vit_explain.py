import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from ViT_Implementation.model import ViT
import os


from Visualization.vit_rollout import VITAttentionRollout
from Visualization.vit_grad_rollout import VITAttentionGradRollout

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Our model path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def main(image_path, model_path=None, use_cuda=False, category_index=None, head_fusion="max", discard_ratio=0.9):
    model = None
    attention_layer_name = 'attn_drop'
    SIZE = 224
    output_filename = "output.png"
    if model_path:
        print("Using our model")
        attention_layer_name = 'softmax'
        base_name = os.path.basename(model_path)
        file_name_without_extension = os.path.splitext(base_name)[0]
        parts = file_name_without_extension.split("_")
        print(parts)

        output_filename = f"Visualization/outputs/{file_name_without_extension}.png"

        nheads = int(parts[0])
        nblocks = int(parts[1])
        hidden_dim = int(parts[2])
        norm = bool(parts[3])
        res = bool(parts[4])
        pos = parts[5] if len(parts) > 5 else "trig"
        
        model = ViT(
            n_blocks=nblocks,
            hidden_dim=hidden_dim,
            n_heads=nheads,
            n_classes=10,
            res=res,
            layer=norm,
            pos=pos
        )
        model.load_state_dict(torch.load(model_path))
        SIZE = 64
    else:
        print("Using facebook model")
        model = torch.hub.load('facebookresearch/deit:main', 
            'deit_tiny_patch16_224', pretrained=True)
        
    model.eval()

    if use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(image_path)
    img = img.resize((SIZE, SIZE))
    img = img.convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    if use_cuda:
        input_tensor = input_tensor.cuda()

    if category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = VITAttentionRollout(model, head_fusion=head_fusion, 
            discard_ratio=args.discard_ratio)
        mask = attention_rollout(input_tensor)
        name = "attention_rollout_{:.3f}_{}.png".format(discard_ratio, head_fusion)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=discard_ratio, attention_layer_name=attention_layer_name)
        mask = grad_rollout(input_tensor, category_index)
        name = "grad_rollout_{}_{:.3f}_{}.png".format(category_index,
            discard_ratio, head_fusion)


    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    #cv2.imshow("Input Image", np_img)
    #cv2.imshow(name, mask)
    cv2.imwrite(output_filename, np_img)
    cv2.imwrite(output_filename, mask)
    print("wrote file to ", output_filename)
    #cv2.waitKey(-1)
    return mask

if __name__ == '__main__':
    args = get_args()
    main(args.model_path, args.use_cuda, args.category_index, args.head_fusion, args.discard_ratio)

