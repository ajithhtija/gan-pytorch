import torch
import torch.nn.functional as F
import math
import numpy as np
# to_tensor = transforms.ToTensor()
def psnr(img1, img2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Supports tensors or NumPy arrays, in [0, 1] or [0, 255].
    """
    original = np.array(img1, dtype=np.float32)
    compressed = np.array(img2, dtype=np.float32)
    # Auto-normalize if needed
    # print(original)
    # print(compressed)
    if original.max() > 1: original /= 255.0
    if compressed.max() > 1: compressed /= 255.0
    print(original.shape,compressed.shape)
    if original.shape != compressed.shape:
        raise ValueError("Images must have the same dimensions and channels")

    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * np.log10(1/ np.sqrt(mse))  # max_pixel = 1.0
def split2(dataset, size, h, w):
    """
    Split a batch of images into smaller patches of size 256x256.
    dataset: PyTorch tensor of shape (batch_size, channels, height, width).
    size: Batch size.
    h, w: Dimensions of the input images.
    """
    nsize1, nsize2 = 256, 256
    patches = []

    for i in range(size):
        img = dataset[i]
        for ii in range(0, h, nsize1):
            for iii in range(0, w, nsize2):
                patch = img[:, ii:ii + nsize1, iii:iii + nsize2]
                patches.append(patch)

    return torch.stack(patches)


def merge_image2(splitted_images, h, w):
    """
    Merge a list of patches back into a full image.
    splitted_images: PyTorch tensor of shape (num_patches, channels, 256, 256).
    h, w: Original dimensions of the image to reconstruct.
    """
    image = torch.zeros((1, h, w), device=splitted_images.device)
    nsize1, nsize2 = 256, 256
    ind = 0

    for ii in range(0, h, nsize1):
        for iii in range(0, w, nsize2):
            image[:, ii:ii + nsize1, iii:iii + nsize2] = splitted_images[ind]
            ind += 1

    return image


import torch.nn.functional as F

import torch.nn.functional as F

def getPatches(watermarked_image, clean_image, mystride):
    """
    Extract 256x256 grayscale patches from torch tensors of shape [1, H, W].
    Returns: tensors of shape [N, 1, 256, 256]
    """
    # print(watermarked_image.shape, clean_image.shape)
    # watermarked_image = to_tensor(watermarked_image)
    # clean_image = to_tensor(clean_image)
    patch_size = 256
    wm_patches = []
    clean_patches = []
    
    # Ensure shape is [1, H, W]
    if watermarked_image.dim() == 2:
        watermarked_image = watermarked_image.unsqueeze(0)
    if clean_image.dim() == 2:
        clean_image = clean_image.unsqueeze(0)

    def pad_to_multiple(img):
        c, h, w = img.shape
        pad_h = ((h + patch_size - 1) // patch_size) * patch_size
        pad_w = ((w + patch_size - 1) // patch_size) * patch_size
        pad_bottom = pad_h - h
        pad_right = pad_w - w
        return F.pad(img, (0, pad_right, 0, pad_bottom), mode='constant', value=1.0)

    # Keep as torch tensors (not numpy)
    watermarked_image = pad_to_multiple(watermarked_image)
    clean_image = pad_to_multiple(clean_image)

    _, h, w = watermarked_image.shape

    for y in range(0, h - patch_size + 1, mystride):
        for x in range(0, w - patch_size + 1, mystride):
            wm_patch = watermarked_image[:, y:y+patch_size, x:x+patch_size]
            clean_patch = clean_image[:, y:y+patch_size, x:x+patch_size]

            if wm_patch.shape[-2:] == (256, 256):
                wm_patches.append(wm_patch)
                clean_patches.append(clean_patch)

    wm_patches = torch.stack(wm_patches)  # [N, 1, 256, 256]
    clean_patches = torch.stack(clean_patches)

    return wm_patches, clean_patches

