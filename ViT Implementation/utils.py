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
