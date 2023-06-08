from vit_explain import main as get_attention_mask
from Segmentation.segment import segment_image as get_segmentation_masks
import sys

if __name__ == "__main__":
    image_path = sys.argv[1]
    model_path = sys.argv[2]

    attention_mask = get_attention_mask(head_fusion="max", discard_ratio=0.9, category_index=0, model_path=model_path, )
    segmentation_mask = get_segmentation_masks(image_path)[0]
    print(attention_mask)
    print(segmentation_mask)