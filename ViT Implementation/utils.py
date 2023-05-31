import torch


# Converts an image of dims [N, C, H, W] to patches of dims [N, n_patches^2, H * W * C / n_patches^2]
def image_to_patches(image, n_patches):
    # patch_size is inferred as H/n_patches or W/n_patches
    patch_size = image.size(2) // n_patches
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # Rearrange to [N, n_patches*n_patches, C * patch_size * patch_size]
    patches = patches.view(patches.size(0), -1, patches.size(-1) * patches.size(-2) * patches.size(-3))
    return patches


# Code from https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def get_positional_embeddings(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return pos_encoding

